"""Task registry for the RelBench ``rel-event`` dataset.

Mirrors the ``rel-f1`` task registry: each :class:`TaskSpec` records everything
downstream code needs to (a) pick the right stitched Parquet, (b) mark the
correct RelFlow leaf ``target=True``, and (c) monitor the leaderboard metric
that RelBench uses for scoring
(https://relbench.stanford.edu/datasets/rel-event/).

Root shapes
-----------
* ``event_interest`` — one row per ``event_interest`` record. Serves the
  autocomplete tasks ``event_interest-interested`` and
  ``event_interest-not_interested``. Both labels live inline on the row (as
  ``interested`` / ``not_interested``); RelBench asks you to drop the other
  column to avoid trivially leaking the answer.
* ``user`` — one row per ``(user, as_of)`` observation drawn from the
  RelBench task train / val / test tables. Serves the forecasting tasks
  ``user-repeat``, ``user-ignore``, ``user-attendance`` and the autocomplete
  task ``users-birthyear``. Each parquet carries its own label column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# Copied from ``dataset.val_timestamp`` / ``dataset.test_timestamp`` so schema-
# building code does not have to re-load the RelBench dataset just to know
# where the temporal splits are.
VALID_TS = datetime(2012, 11, 21)
TEST_TS = datetime(2012, 11, 29)

RootShape = Literal["user", "event_interest"]
TaskType = Literal["regression", "binary"]


@dataclass(frozen=True)
class TaskSpec:
    """One RelBench task and its RelFlow-side wiring."""

    name: str
    root_shape: RootShape
    root_name: str
    parquet_stem: str
    target_path: str            # slash-separated path from the root
    label_column: str           # source column that carries the label
    task_type: TaskType
    relbench_metric: str
    relflow_metric: str
    monitor_mode: Literal["min", "max"]

    @property
    def target_address(self) -> str:
        return f"{self.root_name}/{self.target_path}"

    def metric_key(self, split: str = "validate", face: str = "content") -> str:
        return f"{self.root_name}:{self.target_path}/{split}:{self.relflow_metric}:{face}"


TASKS: dict[str, TaskSpec] = {
    "user-repeat": TaskSpec(
        name="user-repeat",
        root_shape="user",
        root_name="user",
        parquet_stem="rel_event_user_repeat.parquet",
        target_path="label_repeat",
        label_column="target",
        task_type="binary",
        relbench_metric="AUROC",
        relflow_metric="auc",
        monitor_mode="max",
    ),
    "user-ignore": TaskSpec(
        name="user-ignore",
        root_shape="user",
        root_name="user",
        parquet_stem="rel_event_user_ignore.parquet",
        target_path="label_ignore",
        label_column="target",
        task_type="binary",
        relbench_metric="AUROC",
        relflow_metric="auc",
        monitor_mode="max",
    ),
    "user-attendance": TaskSpec(
        name="user-attendance",
        root_shape="user",
        root_name="user",
        parquet_stem="rel_event_user_attendance.parquet",
        target_path="label_attendance",
        label_column="target",
        task_type="regression",
        relbench_metric="MAE",
        relflow_metric="mae",
        monitor_mode="min",
    ),
    "users-birthyear": TaskSpec(
        name="users-birthyear",
        root_shape="user",
        root_name="user",
        parquet_stem="rel_event_users_birthyear.parquet",
        target_path="birthyear",
        label_column="birthyear",
        task_type="regression",
        relbench_metric="MAE",
        relflow_metric="mae",
        monitor_mode="min",
    ),
    "event_interest-interested": TaskSpec(
        name="event_interest-interested",
        root_shape="event_interest",
        root_name="event_interest",
        parquet_stem="rel_event_event_interest_rooted.parquet",
        target_path="interested",
        label_column="interested",
        task_type="binary",
        relbench_metric="AUROC",
        relflow_metric="auc",
        monitor_mode="max",
    ),
    "event_interest-not_interested": TaskSpec(
        name="event_interest-not_interested",
        root_shape="event_interest",
        root_name="event_interest",
        parquet_stem="rel_event_event_interest_rooted.parquet",
        target_path="not_interested",
        label_column="not_interested",
        task_type="binary",
        relbench_metric="AUROC",
        relflow_metric="auc",
        monitor_mode="max",
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(
            f"Unknown rel-event task {name!r}. Known: {sorted(TASKS)}"
        )
    return TASKS[name]
