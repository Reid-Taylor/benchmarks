"""Stitch the RelBench ``rel-f1`` database into a race-rooted nested Parquet file.

The output is one observation per race, mirroring RelFlow's "order + line_items"
pattern (https://relflow.ai/core-concepts/data-flow.html): race attributes and
circuit descriptors sit at the root; qualifying entries, results, driver and
constructor standings, and per-race constructor results are nested lists of
records. A downstream ``relflow.Model`` schema binds these fields directly,
e.g. ``results=rf.Branch(length=..., grid=rf.Number, position=rf.Number, ...)``.

Run::

    python f1_stitching.py --output rel_f1_stitched.parquet

Then load the parquet with ``polars.read_parquet`` and pass it to
``rf.PolarsDataModule`` when training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from relbench.datasets import get_dataset

# Race is the natural transactional root: every child table carries a raceId FK.
ROOT_TABLE = "races"
ROOT_KEY = "raceId"

# Child tables to nest under each race, keyed by the branch name that will
# appear in the output (and in the RelFlow schema).
CHILD_TABLES: dict[str, str] = {
    "results": "results",
    "qualifying": "qualifying",
    "driver_standings": "standings",
    "constructor_results": "constructor_results",
    "constructor_standings": "constructor_standings",
}


def _prefix(df: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    """Prefix every column except join keys so joined columns don't collide."""
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in keep})


def _enrich(df: pd.DataFrame, drivers: pd.DataFrame, constructors: pd.DataFrame) -> pd.DataFrame:
    """Fold driver / constructor descriptors into rows that reference them."""
    if "driverId" in df.columns:
        df = df.merge(drivers, on="driverId", how="left")
    if "constructorId" in df.columns:
        df = df.merge(constructors, on="constructorId", how="left")
    return df


def _nest(df: pd.DataFrame, branch_name: str) -> pd.DataFrame:
    """Group ``df`` by ``ROOT_KEY`` and collect each group as a list of dicts."""
    payload_cols = [c for c in df.columns if c != ROOT_KEY]
    records = df[payload_cols].to_dict(orient="records")
    grouped = (
        pd.DataFrame({ROOT_KEY: df[ROOT_KEY].to_numpy(), "_record": records})
        .groupby(ROOT_KEY, sort=False)["_record"]
        .apply(list)
        .reset_index(name=branch_name)
    )
    return grouped


def stitch(db) -> pd.DataFrame:
    tables = {name: table.df for name, table in db.table_dict.items()}

    drivers = _prefix(tables["drivers"], "driver", keep={"driverId"})
    constructors = _prefix(tables["constructors"], "constructor", keep={"constructorId"})
    circuits = _prefix(tables["circuits"], "circuit", keep={"circuitId"})

    races = tables[ROOT_TABLE].merge(circuits, on="circuitId", how="left")

    stitched = races
    for branch_name, source_table in CHILD_TABLES.items():
        child = _enrich(tables[source_table], drivers, constructors)
        if ROOT_KEY not in child.columns:
            raise KeyError(f"Table {source_table!r} has no {ROOT_KEY} column to nest on.")
        nested = _nest(child, branch_name)
        stitched = stitched.merge(nested, on=ROOT_KEY, how="left")
        # Races with no children become [] rather than NaN so the schema sees a valid list.
        stitched[branch_name] = stitched[branch_name].apply(
            lambda v: v if isinstance(v, list) else []
        )

    return stitched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rel_f1_stitched.parquet"),
        help="Destination parquet path (default: %(default)s).",
    )
    args = parser.parse_args()

    dataset = get_dataset("rel-f1")
    db = dataset.get_db()
    stitched = stitch(db)

    stitched.to_parquet(args.output, engine="pyarrow", index=False)

    branch_summary = ", ".join(
        f"{name}=avg {stitched[name].map(len).mean():.1f}" for name in CHILD_TABLES
    )
    print(
        f"Wrote {len(stitched):,} race observations "
        f"({len(stitched.columns)} root columns) to {args.output}\n"
        f"  branches: {branch_summary}"
    )


if __name__ == "__main__":
    main()