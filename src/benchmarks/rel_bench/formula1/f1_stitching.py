"""Stitch the RelBench ``rel-f1`` database into one nested Parquet per task.

There are two observation shapes, matching the two families of official rel-f1
tasks (https://relbench.stanford.edu/dataset_info/rel-f1/):

Race-rooted (`autocomplete` tasks: ``results-position``, ``qualifying-position``)
    One row per race. Every child table (results, qualifying) is nested as a
    list of records under its branch name. Driver, constructor, and circuit
    descriptors are folded into each row so downstream schemas can bind either
    the raw ID (via ``rf.Entity``) or the descriptor (via ``rf.Category`` /
    ``rf.Number``) without a re-join. The RelBench autocomplete task label
    already lives inside its branch (``results.position`` or
    ``qualifying.position``); the training script only needs to mark that leaf
    ``target=True``.

Driver-rooted (`forecasting` tasks: ``driver-dnf``, ``driver-top3``,
``driver-position``)
    One row per ``(driverId, as_of)`` entry from RelBench's official task
    train / val / test tables. The row carries the RelBench label, an
    ``as_of`` timestamp, driver descriptors, a ``split`` marker (``train`` /
    ``validate`` / ``test``), and a ``recent_results`` branch containing the
    driver's most recent race results strictly before ``as_of``. This keeps
    the observation temporally leakage-safe and mirrors how RelBench itself
    scores each row.

Usage
-----

Build every parquet (race-rooted + one per forecast task)::

    python -m benchmarks.rel_bench.formula1.f1_stitching --output-dir data/

Build a single task's parquet::

    python -m benchmarks.rel_bench.formula1.f1_stitching \\
        --task driver-dnf --output-dir data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from .tasks import TASKS, TaskSpec

RACE_ROOTED_STEM = "rel_f1_race_rooted.parquet"

RACE_CHILDREN: dict[str, str] = {
    "results": "results",
    "qualifying": "qualifying",
}

DRIVER_HISTORY_LENGTH = 30


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prefix(df: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    return df.rename(
        columns={c: f"{prefix}_{c}" for c in df.columns if c not in keep}
    )


def _enrich(
    df: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
) -> pd.DataFrame:
    if "driverId" in df.columns:
        df = df.merge(drivers, on="driverId", how="left")
    if "constructorId" in df.columns:
        df = df.merge(constructors, on="constructorId", how="left")
    return df


def _nest(df: pd.DataFrame, key: str, branch: str) -> pd.DataFrame:
    payload_cols = [c for c in df.columns if c != key]
    records = df[payload_cols].to_dict(orient="records")
    grouped = (
        pd.DataFrame({key: df[key].to_numpy(), "_record": records})
        .groupby(key, sort=False)["_record"]
        .apply(list)
        .reset_index(name=branch)
    )
    return grouped


def _split_from_date(dates: pd.Series, valid_ts, test_ts) -> pd.Series:
    return pd.Series(
        pd.cut(
            dates,
            bins=[pd.Timestamp.min, valid_ts, test_ts, pd.Timestamp.max],
            labels=["train", "validate", "test"],
            right=False,
        )
    ).astype(str)


# ---------------------------------------------------------------------------
# Race-rooted parquet
# ---------------------------------------------------------------------------


def build_race_rooted(db, valid_ts, test_ts) -> pd.DataFrame:
    tables = {name: table.df for name, table in db.table_dict.items()}

    drivers = _prefix(tables["drivers"], "driver", keep={"driverId"})
    constructors = _prefix(
        tables["constructors"], "constructor", keep={"constructorId"}
    )
    circuits = _prefix(tables["circuits"], "circuit", keep={"circuitId"})

    races = tables["races"].merge(circuits, on="circuitId", how="left")

    stitched = races
    for branch_name, source_table in RACE_CHILDREN.items():
        child = _enrich(tables[source_table], drivers, constructors)
        if "raceId" not in child.columns:
            raise KeyError(f"{source_table!r} has no raceId column")
        nested = _nest(child, key="raceId", branch=branch_name)
        stitched = stitched.merge(nested, on="raceId", how="left")
        stitched[branch_name] = stitched[branch_name].apply(
            lambda v: v if isinstance(v, list) else []
        )

    stitched["split"] = _split_from_date(stitched["date"], valid_ts, test_ts)
    return stitched


# ---------------------------------------------------------------------------
# Driver-rooted parquets (forecasting tasks)
# ---------------------------------------------------------------------------


def _build_history_index(db) -> pd.DataFrame:
    """One row per (driverId, raceId) with the features we want to include
    in a driver's history branch. Sorted by driverId, date so a windowed
    tail-lookup is a single pandas ``groupby``.
    """
    tables = {name: table.df for name, table in db.table_dict.items()}

    drivers = _prefix(tables["drivers"], "driver", keep={"driverId"})
    constructors = _prefix(
        tables["constructors"], "constructor", keep={"constructorId"}
    )
    circuits = _prefix(tables["circuits"], "circuit", keep={"circuitId"})

    results = tables["results"].merge(constructors, on="constructorId", how="left")
    races = tables["races"].merge(circuits, on="circuitId", how="left")

    history = results.merge(
        races[
            [
                "raceId",
                "date",
                "year",
                "round",
                "name",
                "circuit_country",
                "circuit_lat",
                "circuit_lng",
                "circuit_alt",
            ]
        ].rename(columns={"date": "race_date", "name": "race_name"}),
        on="raceId",
        how="left",
    )

    keep_cols = [
        "driverId",
        "race_date",
        "raceId",
        "year",
        "round",
        "race_name",
        "circuit_country",
        "circuit_lat",
        "circuit_lng",
        "circuit_alt",
        "constructorId",
        "constructor_nationality",
        "grid",
        "position",
        "positionOrder",
        "points",
        "laps",
        "statusId",
    ]
    history = history[[c for c in keep_cols if c in history.columns]].copy()
    history = history.sort_values(["driverId", "race_date"]).reset_index(drop=True)

    # Cache driver descriptors so we can join them onto each observation.
    history = history.merge(drivers, on="driverId", how="left")
    return history


def _gather_recent(
    history: pd.DataFrame,
    driver_id,
    as_of: pd.Timestamp,
    length: int,
) -> list[dict]:
    sub = history[
        (history["driverId"] == driver_id) & (history["race_date"] < as_of)
    ]
    if sub.empty:
        return []
    tail = sub.tail(length)
    # Every row in ``tail`` shares the same driver descriptors; strip them
    # from the branch payload so each result record stays compact.
    drop = {
        "driverId",
        "driver_forename",
        "driver_surname",
        "driver_nationality",
        "driver_dob",
        "driver_code",
        "driver_number",
        "driver_driverRef",
        "driver_url",
    }
    payload = tail.drop(columns=[c for c in drop if c in tail.columns])
    return payload.to_dict(orient="records")


def build_driver_rooted_for_task(
    db,
    task_name: str,
    length: int = DRIVER_HISTORY_LENGTH,
) -> pd.DataFrame:
    """One row per (driverId, timestamp) from the RelBench task train / val /
    test tables, with the driver's most recent ``length`` result rows nested
    under ``recent_results`` and the RelBench label copied through under
    ``label_<task>``.

    Test-split rows have ``label`` set to None (RelBench withholds labels)
    but are still emitted so the trained model can produce a submission-ready
    prediction file.
    """
    spec: TaskSpec = TASKS[task_name]
    task = get_task("rel-f1", task_name, download=True)

    history = _build_history_index(db)
    drivers = _prefix(db.table_dict["drivers"].df, "driver", keep={"driverId"})

    frames = []
    for split_name, table_key in (
        ("train", "train"),
        ("validate", "val"),
        ("test", "test"),
    ):
        df = task.get_table(table_key).df.copy()
        df["split"] = split_name
        if spec.label_column not in df.columns:
            df[spec.label_column] = None
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.rename(
        columns={"date": "as_of", spec.label_column: spec.target_path}
    )

    # Attach driver descriptors onto every observation.
    all_rows = all_rows.merge(drivers, on="driverId", how="left")

    all_rows["recent_results"] = [
        _gather_recent(history, driver_id, as_of, length)
        for driver_id, as_of in zip(all_rows["driverId"], all_rows["as_of"])
    ]
    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Destination directory (parquet files land here).",
    )
    parser.add_argument(
        "--task",
        choices=[*TASKS, "race-rooted", "all"],
        default="all",
        help="Build one specific task's parquet, the shared race-rooted "
             "parquet, or every parquet (default).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=DRIVER_HISTORY_LENGTH,
        help="Maximum number of past race entries to nest under "
             "recent_results for each driver-rooted observation.",
    )
    args = parser.parse_args()

    dataset = get_dataset("rel-f1")
    db = dataset.get_db()
    valid_ts = pd.Timestamp(dataset.val_timestamp)
    test_ts = pd.Timestamp(dataset.test_timestamp)

    if args.task in ("race-rooted", "all") or (
        args.task in TASKS and TASKS[args.task].root_shape == "race"
    ):
        race_df = build_race_rooted(db, valid_ts, test_ts)
        race_path = args.output_dir / RACE_ROOTED_STEM
        _write(race_df, race_path)
        counts = race_df["split"].value_counts().to_dict()
        print(
            f"[race-rooted] {len(race_df):,} rows -> {race_path}  "
            f"splits={counts}"
        )

    forecast_tasks = [
        name for name, spec in TASKS.items() if spec.root_shape == "driver"
    ]
    if args.task == "all":
        chosen = forecast_tasks
    elif args.task in TASKS and TASKS[args.task].root_shape == "driver":
        chosen = [args.task]
    else:
        chosen = []

    for name in chosen:
        df = build_driver_rooted_for_task(db, name, length=args.history_length)
        path = args.output_dir / TASKS[name].parquet_stem
        _write(df, path)
        counts = df["split"].value_counts().to_dict()
        hist_avg = df["recent_results"].map(len).mean()
        print(
            f"[{name}] {len(df):,} rows -> {path}  "
            f"splits={counts}  recent_results avg={hist_avg:.1f}"
        )


if __name__ == "__main__":
    main()
