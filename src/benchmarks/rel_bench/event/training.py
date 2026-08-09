"""Train a RelFlow model on the RelBench ``user-repeat`` task.

``user-repeat`` is a binary forecasting task
(https://relbench.stanford.edu/datasets/rel-event/): predict whether a user
will respond ``yes`` or ``maybe`` to at least one event in the next 7 days,
given they already attended an event in the last 14 days. RelBench scores
this task with **AUROC**.

The script loads the parquet produced by ``stitching.py`` for the chosen task
(currently ``user-repeat`` only), builds a user-rooted RelFlow schema whose
target leaf is ``label_repeat``, and monitors the RelFlow metric key that
corresponds to the RelBench leaderboard metric.

Interpretable logging
---------------------
At startup the script prints:

* the RelBench task name and its official evaluation metric;
* the RelFlow schema address of the target leaf;
* the exact ``trainer.callback_metrics`` keys emitted per split.

Every printed key can be pasted directly into ``ModelCheckpoint(monitor=...)``
or a wandb chart. RelFlow's ``rf.Boolean(target=True)`` tracks ``auc``
(BinaryAUROC) in-training under an address-qualified key -- that number IS
the RelBench leaderboard metric on the split whose labels are available.
After training, the exact RelBench-official score is re-computed by handing
the predicted ``P(True)`` to ``task.evaluate(...)`` from ``relbench`` on
every split whose labels are available.

Example
-------
::

    python -m benchmarks.rel_bench.event.training \\
        --task user-repeat --parquet-dir data/ --max-epochs 20
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger

from .tasks import TASKS, TaskSpec, get_task_spec

# shared Unix account: force wandb to use MY key, never ~/.netrc
assert os.environ.get("WANDB_API_KEY"), "source ~/reid/wandb.env before running"
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)


# Only user-repeat is implemented in this file. The registry contains every
# rel-event task so the CLI vocabulary and evaluation code stay honest; adding
# another task means writing a matching schema builder below and expanding
# SUPPORTED_TASKS.
SUPPORTED_TASKS: tuple[str, ...] = ("user-repeat",)

# ---------------------------------------------------------------------------
# Bounded vocabularies (soft ceilings sized above observed rel-event
# cardinalities: 64 locales, 2 693 non-null locations, 2 genders, 202
# countries, 772 states, 28 764 cities, 37 143 users, 2.46M events, 4
# event_attendees statuses).
# ---------------------------------------------------------------------------
N_USERS = 65_536
N_EVENTS = 4_194_304
N_LOCALES = 128
N_GENDERS = 4
N_LOCATIONS = 4096
N_COUNTRIES = 256
N_STATES = 1024
N_CITIES = 32_768
N_STATUS = 8
N_INTEREST_HISTORY = 32
N_ATTENDANCE_HISTORY = 32
N_FRIENDS = 64


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------


def _target(spec: TaskSpec):
    """Return the RelFlow constructor for the supervised leaf of ``spec``."""
    if spec.task_type == "regression":
        return rf.Number(target=True)
    return rf.Boolean(target=True)


def build_user_repeat_schema(spec: TaskSpec) -> rf.Model:
    """User-rooted schema for the ``user-repeat`` forecasting task.

    Each observation is one ``(user_id, as_of)`` pair with the RelBench label
    at ``label_repeat`` and three context branches:

    * ``recent_interests`` — last N event_interest rows strictly before as_of;
    * ``recent_attendance`` — last N event_attendees rows strictly before as_of;
    * ``friends`` — untimed snapshot of the user's social graph.
    """
    recent_interests = rf.Branch(
        length=N_INTEREST_HISTORY,
        event_id=rf.Entity(),
        event_owner=rf.Entity(),
        invited=rf.Number(),
        timestamp=rf.DateParts(
            dateparts=["month_of_year", "day_of_week", "week_of_year"],
        ),
        interested=rf.Number(),
        not_interested=rf.Number(),
        event_start_time=rf.DateParts(
            dateparts=["month_of_year", "day_of_week", "week_of_year"],
        ),
        event_city=rf.Category(size=N_CITIES),
        event_state=rf.Category(size=N_STATES),
        event_country=rf.Category(size=N_COUNTRIES),
        event_lat=rf.Number(),
        event_lng=rf.Number(),
    )
    recent_attendance = rf.Branch(
        length=N_ATTENDANCE_HISTORY,
        event_id=rf.Entity(),
        event_owner=rf.Entity(),
        status=rf.Category(size=N_STATUS),
        start_time=rf.DateParts(
            dateparts=["month_of_year", "day_of_week", "week_of_year"],
        ),
        event_start_time=rf.DateParts(
            dateparts=["month_of_year", "day_of_week", "week_of_year"],
        ),
        event_city=rf.Category(size=N_CITIES),
        event_state=rf.Category(size=N_STATES),
        event_country=rf.Category(size=N_COUNTRIES),
        event_lat=rf.Number(),
        event_lng=rf.Number(),
    )
    friends = rf.Branch(
        length=N_FRIENDS,
        friend=rf.Entity(),
        friend_locale=rf.Category(size=N_LOCALES),
        friend_birthyear=rf.Number(),
        friend_gender=rf.Category(size=N_GENDERS),
        friend_joinedAt=rf.DateParts(
            dateparts=["month_of_year", "day_of_week", "week_of_year"],
        ),
        friend_location=rf.Category(size=N_LOCATIONS),
        friend_timezone=rf.Number(),
    )

    return rf.Model(
        name=spec.root_name,
        d_model=256,
        n_layers=8,
        n_heads=8,
        batch_size=128,
        embed=True,
        optimizer=lambda module: torch.optim.AdamW(
            module.parameters(), lr=1e-3, weight_decay=1e-4
        ),
        user_id=rf.Category(size=N_USERS),
        as_of=rf.DateParts(
            dateparts=["month_of_year", "day_of_week", "week_of_year"],
            query="[*].as_of",
        ),
        user_locale=rf.Category(size=N_LOCALES),
        user_birthyear=rf.Number(),
        user_gender=rf.Category(size=N_GENDERS),
        user_joinedAt=rf.DateParts(
            dateparts=["month_of_year", "day_of_week", "week_of_year"],
        ),
        user_location=rf.Category(size=N_LOCATIONS),
        user_timezone=rf.Number(),
        recent_interests=recent_interests,
        recent_attendance=recent_attendance,
        friends=friends,
        **{spec.target_path: _target(spec)},
    )


def build_model_for(spec: TaskSpec) -> rf.Model:
    if spec.name == "user-repeat":
        return build_user_repeat_schema(spec)
    raise NotImplementedError(
        f"training schema for task {spec.name!r} is not implemented yet; "
        f"only {SUPPORTED_TASKS} are supported."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_splits(
    parquet_dir: Path, spec: TaskSpec
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    path = parquet_dir / spec.parquet_stem
    frame = pl.read_parquet(path)
    if "split" not in frame.columns:
        raise KeyError(
            f"{path} is missing a 'split' column — regenerate it with "
            "stitching.py."
        )
    if spec.task_type == "binary" and spec.target_path in frame.columns:
        # rf.Boolean rejects int labels; cast 0/1 -> False/True at the root.
        # Rows whose label is null (RelBench test split) stay null.
        frame = frame.with_columns(pl.col(spec.target_path).cast(pl.Boolean))
    return (
        frame.filter(pl.col("split") == "train"),
        frame.filter(pl.col("split") == "validate"),
        frame.filter(pl.col("split") == "test"),
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logger(spec: TaskSpec, tag: str) -> Logger | bool:
    """Return a wandb logger if credentials are present, else Lightning's
    default logger."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("[logger] WANDB_API_KEY not set — using Lightning's default logger.")
        return True
    try:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        print("[logger] wandb not installed — using Lightning's default logger.")
        return True
    wandb.login(key=api_key)
    return WandbLogger(
        entity=os.environ.get("WANDB_ENTITY", "rebridgers-independent"),
        project=os.environ.get("WANDB_PROJECT", "rel-bench"),
        name=f"{spec.name}-{tag}",
        tags=[spec.name, spec.root_shape, spec.task_type],
        config={
            "task": spec.name,
            "root_shape": spec.root_shape,
            "target_address": spec.target_address,
            "relbench_metric": spec.relbench_metric,
            "relflow_metric": spec.relflow_metric,
        },
    )


def _announce(spec: TaskSpec, monitor_key: str) -> None:
    print("=" * 72)
    print(f"  RelBench task : {spec.name}")
    print(f"  RelBench score: {spec.relbench_metric}  (leaderboard metric)")
    print(f"  Root shape    : {spec.root_shape}  (root name = {spec.root_name!r})")
    print(f"  Target leaf   : {spec.target_address}  ({spec.task_type})")
    print(f"  Monitoring    : {monitor_key}  (mode={spec.monitor_mode})")
    print("  Split metrics you can chart:")
    for split in ("train", "validate", "test"):
        print(f"    - {spec.metric_key(split)}")
        print(f"    - loss/{split}")
    print("=" * 72)


def _dump_final_metrics(spec: TaskSpec, trainer: lit.Trainer) -> None:
    print("\n[final callback_metrics]")
    for key, value in sorted(trainer.callback_metrics.items()):
        marker = "  * " if spec.target_address in key else "    "
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            fvalue = float("nan")
        print(f"{marker}{key} = {fvalue:.5f}")


# ---------------------------------------------------------------------------
# RelBench-official post-fit evaluation
# ---------------------------------------------------------------------------

# RelBench splits are named 'train'/'val'/'test'; our parquet uses 'validate'.
_RELBENCH_SPLIT = {"train": "train", "validate": "val", "test": "test"}


def _predict_scores(
    spec: TaskSpec,
    model: rf.Model,
    frame: pl.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """Run ``model.predict`` in batches and return one score per row.

    Binary tasks return ``P(True)`` (from the ``rf.Boolean`` target). Regression
    tasks return the point prediction. Both are the exact inputs
    ``relbench.tasks.Task.evaluate`` expects.
    """
    rows = frame.to_dicts()
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        out = model.predict(batch)
        content = out[spec.target_address]["content"]
        if spec.task_type == "binary":
            probs = content["probability"]
        else:
            probs = content
        scores.extend(float(x) for x in probs)
    return np.asarray(scores, dtype=np.float64)


def relbench_evaluate(
    spec: TaskSpec,
    model: rf.Model,
    frame: pl.DataFrame,
    split: str,
    *,
    batch_size: int = 64,
) -> dict[str, float] | None:
    """Compute RelBench-official leaderboard metrics on a labeled split.

    For user-rooted tasks, alignment is by ``(user, timestamp)`` — the
    RelBench task table's entity + time columns. Our parquet stores those as
    ``user_id`` and ``as_of``.
    """
    if len(frame) == 0:
        print(f"[relbench-eval] no rows in {split!r} split, skipping.")
        return None

    from relbench.tasks import get_task

    task = get_task("rel-event", spec.name)
    relbench_split = _RELBENCH_SPLIT.get(split, split)
    target_table = task.get_table(relbench_split)
    labels_df = target_table.df
    entity_col = task.entity_col
    time_col = task.time_col

    if task.target_col not in labels_df.columns:
        # RelBench ships test tables without labels; predictions there must be
        # submitted to the leaderboard.
        print(
            f"[relbench-eval] {spec.name} {relbench_split!r} split has no "
            "labels (must be submitted to the RelBench leaderboard). Skipping."
        )
        return None

    scores = _predict_scores(spec, model, frame, batch_size=batch_size)
    frame_pd = frame.select(["user_id", "as_of"]).to_pandas()
    frame_pd["score"] = scores
    key_to_score: dict[tuple[int, object], float] = {
        (int(row.user_id), row.as_of): float(row.score)
        for row in frame_pd.itertuples(index=False)
    }

    pred = np.full(len(labels_df), np.nan, dtype=np.float64)
    for i, row in enumerate(labels_df.itertuples(index=False)):
        pred[i] = key_to_score.get(
            (int(getattr(row, entity_col)), getattr(row, time_col)),
            np.nan,
        )
    missing = int(np.isnan(pred).sum())
    if missing:
        print(
            f"[relbench-eval] warning: {missing}/{len(pred)} rows in the "
            f"RelBench {relbench_split!r} table had no matching parquet row; "
            "filling with 0.5 for binary / mean for regression."
        )
        fill = 0.5 if spec.task_type == "binary" else float(np.nanmean(pred))
        pred = np.where(np.isnan(pred), fill, pred)

    return task.evaluate(pred, target_table)


def _print_relbench_metrics(
    spec: TaskSpec,
    split: str,
    metrics: dict[str, float],
) -> None:
    official = spec.relbench_metric.lower().replace("auroc", "roc_auc")
    print(
        f"\n[relbench-eval] {spec.name} on RelBench {split!r} split "
        f"(official metric = {spec.relbench_metric}):"
    )
    for key, value in sorted(metrics.items()):
        marker = "  * " if key.lower() == official else "    "
        print(f"{marker}{key} = {float(value):.5f}")


def _log_relbench_metrics(
    logger: Logger | bool,
    spec: TaskSpec,
    split: str,
    metrics: dict[str, float],
) -> None:
    if not isinstance(logger, Logger):
        return
    payload = {
        f"relbench/{spec.name}/{split}/{key}": float(value)
        for key, value in metrics.items()
    }
    payload[f"relbench/{spec.name}/{split}/official"] = float(
        metrics.get(
            spec.relbench_metric.lower().replace("auroc", "roc_auc"),
            float("nan"),
        )
    )
    logger.log_metrics(payload)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        default="user-repeat",
        help="Which official RelBench rel-event task to train. Currently "
             f"only {SUPPORTED_TASKS} is supported.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the parquets produced by stitching.py.",
    )
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Where to write the compact RelFlow artifact. "
             "Default: rel_event_<task>.pt inside the checkpoint dir.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    spec = get_task_spec(args.task)
    if spec.name not in SUPPORTED_TASKS:
        raise SystemExit(
            f"task {spec.name!r} is registered but not yet implemented in "
            f"this training script (supported: {SUPPORTED_TASKS})."
        )

    train, validate, test = load_splits(args.parquet_dir, spec)
    print(
        f"[data] task={spec.name}  train={len(train):,}  "
        f"validate={len(validate):,}  test={len(test):,}"
    )

    model = build_model_for(spec)
    if args.batch_size is not None:
        model.batch_size = args.batch_size

    monitor_key = spec.metric_key("validate")
    _announce(spec, monitor_key)

    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="loss/validate",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=f"{spec.name}-{{epoch:02d}}-{{loss/validate:.4f}}",
    )

    tag = datetime.now().strftime("%Y-%m-%d-%H%M")
    logger = _configure_logger(spec, tag)

    datamodule = rf.PolarsDataModule(
        model=model,
        train=train,
        validate=validate,
        test=test if len(test) else None,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    trainer = lit.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="bf16",
        devices="auto",
        callbacks=[checkpoint],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model=model, datamodule=datamodule)

    if len(test):
        trainer.test(model=model, datamodule=datamodule, verbose=True)

    _dump_final_metrics(spec, trainer)

    val_metrics = relbench_evaluate(
        spec, model, validate, split="validate", batch_size=max(model.batch_size, 32)
    )
    if val_metrics is not None:
        _print_relbench_metrics(spec, "val", val_metrics)
        _log_relbench_metrics(logger, spec, "val", val_metrics)

    if len(test):
        test_metrics = relbench_evaluate(
            spec, model, test, split="test", batch_size=max(model.batch_size, 32)
        )
        if test_metrics is not None:
            _print_relbench_metrics(spec, "test", test_metrics)
            _log_relbench_metrics(logger, spec, "test", test_metrics)

    artifact_path = args.artifact or (
        args.checkpoint_dir / f"rel_event_{spec.name.replace('-', '_')}.pt"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)
    print(f"[artifact] saved RelFlow model to {artifact_path}")


if __name__ == "__main__":
    main()
