"""Task registry for the RelBench ``rel-f1`` dataset.

Each :class:`TaskSpec` records everything the training script needs to select
the right stitched Parquet, wire up the correct ``target=True`` leaf in the
RelFlow schema, and monitor the metric that RelBench uses on its leaderboard
(https://relbench.stanford.edu/dataset_info/rel-f1/).

RelFlow emits address-qualified metrics of the form
``{root_name}:{path}/{split}:{metric}:content`` (see
https://relflow.ai/guides/lightning.html#log-and-inspect-metrics), so the keys
returned by :meth:`TaskSpec.metric_key` are exactly what appears in
``trainer.callback_metrics`` and wandb.

Notes on RelBench alignment
---------------------------
* Regression targets (``results-position``, ``qualifying-position``,
  ``driver-position``) are ``rf.Number(target=True)``; RelFlow logs ``mae`` in
  the original value scale, matching RelBench's regression metric.
* Binary targets (``driver-dnf``, ``driver-top3``) are
  ``rf.Boolean(target=True)``. RelFlow logs ``auc`` (BinaryAUROC) in-training,
  which matches the RelBench leaderboard metric; the exact leaderboard score is
  re-computed after training by handing the predicted P(True) to
  ``task.evaluate(...)`` from ``relbench``.
* ``driver-circuit-compete`` (link prediction / MAP) is deliberately omitted
  here — it requires an ``rf.Set`` target over the circuit vocabulary and a
  bespoke evaluation loop, which is a separate design pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# Copied from ``dataset.val_timestamp`` / ``dataset.test_timestamp`` so schema-
# building code does not have to re-load the RelBench dataset just to know
# where the temporal splits are.
VALID_TS = datetime(2005, 1, 1)
TEST_TS = datetime(2010, 1, 1)

RootShape = Literal["race", "driver"]
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
    "results-position": TaskSpec(
        name="results-position",
        root_shape="race",
        root_name="race",
        parquet_stem="rel_f1_race_rooted.parquet",
        target_path="results/position",
        label_column="position",
        task_type="regression",
        relbench_metric="MAE",
        relflow_metric="mae",
        monitor_mode="min",
    ),
    "qualifying-position": TaskSpec(
        name="qualifying-position",
        root_shape="race",
        root_name="race",
        parquet_stem="rel_f1_race_rooted.parquet",
        target_path="qualifying/position",
        label_column="position",
        task_type="regression",
        relbench_metric="MAE",
        relflow_metric="mae",
        monitor_mode="min",
    ),
    "driver-dnf": TaskSpec(
        name="driver-dnf",
        root_shape="driver",
        root_name="driver",
        parquet_stem="rel_f1_driver_dnf.parquet",
        target_path="label_dnf",
        label_column="did_not_finish",
        task_type="binary",
        relbench_metric="AUROC",
        relflow_metric="auc",
        monitor_mode="max",
    ),
    "driver-top3": TaskSpec(
        name="driver-top3",
        root_shape="driver",
        root_name="driver",
        parquet_stem="rel_f1_driver_top3.parquet",
        target_path="label_top3",
        label_column="qualifying",
        task_type="binary",
        relbench_metric="AUROC",
        relflow_metric="auc",
        monitor_mode="max",
    ),
    "driver-position": TaskSpec(
        name="driver-position",
        root_shape="driver",
        root_name="driver",
        parquet_stem="rel_f1_driver_position.parquet",
        target_path="label_position",
        label_column="position",
        task_type="regression",
        relbench_metric="MAE",
        relflow_metric="mae",
        monitor_mode="min",
    ),
}


def get_task_spec(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(
            f"Unknown rel-f1 task {name!r}. Known: {sorted(TASKS)}"
        )
    return TASKS[name]
