"""Stitch the RelBench ``rel-event`` database into one nested Parquet per task.

There are two observation shapes, matching the two families of official
rel-event tasks (https://relbench.stanford.edu/datasets/rel-event/):

Event-interest-rooted (autocomplete: ``event_interest-interested``,
``event_interest-not_interested``)
    One row per record in the ``event_interest`` table. Every user descriptor
    and event descriptor is folded into the row so the downstream schema can
    bind either the raw ID (via ``rf.Entity``) or the descriptor (via
    ``rf.Category`` / ``rf.Number``) without a re-join. Both label columns
    (``interested`` and ``not_interested``) live on the row; the training
    script marks whichever leaf the task cares about with ``target=True`` and
    drops the sibling column to avoid trivial leakage — this is exactly what
    RelBench itself does per task description.

User-rooted (forecasting: ``user-repeat``, ``user-ignore``,
``user-attendance``; autocomplete: ``users-birthyear``)
    One row per ``(userId, as_of)`` entry from RelBench's official task
    train / val / test tables. The row carries the RelBench label, an
    ``as_of`` timestamp, user descriptors, a ``split`` marker, and three
    context branches:

    * ``recent_interests`` — the user's most recent ``event_interest`` rows
      strictly before ``as_of`` (event descriptors folded in).
    * ``recent_attendance`` — the user's most recent ``event_attendees`` rows
      strictly before ``as_of`` (event descriptors folded in).
    * ``friends`` — a bounded snapshot of the user's social graph from
      ``user_friends``, each friend annotated with the friend's user
      descriptors. ``user_friends`` has no timestamp column in RelBench, so
      we treat it as static.

    The temporal ``strictly-before`` cut keeps forecasting rows leakage-safe
    and mirrors how RelBench itself scores each row. For the autocomplete
    ``users-birthyear`` task there is no forecast horizon, so we use
    ``joinedAt`` (RelBench's declared ``time_col`` for that task) as
    ``as_of`` and expose all-time context; the target column is
    ``birthyear``, so the training script drops ``user_birthyear`` from the
    descriptor set for that parquet.

Usage
-----

Build every parquet (event-interest-rooted + one per user-rooted task)::

    python -m benchmarks.rel_bench.event.stitching --output-dir data/

Build a single task's parquet::

    python -m benchmarks.rel_bench.event.stitching \\
        --task user-repeat --output-dir data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from .tasks import TASKS, TaskSpec

EVENT_INTEREST_ROOTED_STEM = "rel_event_event_interest_rooted.parquet"

# History-window sizes for the user-rooted branches. Sized to keep the
# resulting parquet small enough for local iteration while still exposing
# enough context for a model to learn a temporal signal.
INTEREST_HISTORY_LENGTH = 32
ATTENDANCE_HISTORY_LENGTH = 32
FRIENDS_SNAPSHOT_LENGTH = 64

# All words-cluster columns in ``events``. The full events table has 100
# word-cluster columns (``c_1``..``c_100``, ``c_other``); those are heavy and
# only make sense on the event-interest-rooted parquet where each event
# appears at most once per row. Nested history branches drop them.
EVENT_CLUSTER_COLS = [f"c_{i}" for i in range(1, 101)] + ["c_other"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prefix(df: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    return df.rename(
        columns={c: f"{prefix}_{c}" for c in df.columns if c not in keep}
    )


def _split_from_date(dates: pd.Series, valid_ts, test_ts) -> pd.Series:
    return pd.Series(
        pd.cut(
            dates,
            bins=[pd.Timestamp.min, valid_ts, test_ts, pd.Timestamp.max],
            labels=["train", "validate", "test"],
            right=False,
        )
    ).astype(str)


def _user_descriptors(users: pd.DataFrame) -> pd.DataFrame:
    return _prefix(users, "user", keep={"user_id"})


def _event_descriptors(events: pd.DataFrame, cluster_cols: bool) -> pd.DataFrame:
    """Rename event columns with an ``event_`` prefix but keep ``event_id``.

    ``user_id`` on ``events`` is the event owner; we rename it to
    ``event_owner`` (already prefixed, so we skip re-prefixing it). If
    ``cluster_cols`` is False, the 101 word-cluster columns are dropped.
    """
    df = events.rename(columns={"user_id": "event_owner"})
    if not cluster_cols:
        df = df.drop(columns=[c for c in EVENT_CLUSTER_COLS if c in df.columns])
    return _prefix(df, "event", keep={"event_id", "event_owner"})


# ---------------------------------------------------------------------------
# Event-interest-rooted parquet
# ---------------------------------------------------------------------------


def build_event_interest_rooted(db, valid_ts, test_ts) -> pd.DataFrame:
    """One row per ``event_interest`` record with user + event descriptors
    folded in. Both labels (``interested`` and ``not_interested``) stay
    inline; the downstream schema drops the sibling column to avoid trivial
    leakage per RelBench task description.
    """
    tables = {name: table.df for name, table in db.table_dict.items()}

    ei = tables["event_interest"].copy()
    ei = ei.rename(columns={"user": "user_id", "event": "event_id"})
    # RelBench's autocomplete task table keys back into event_interest with a
    # ``primary_key`` column that is just this row's position; surface it so
    # downstream code can align RelBench's task rows with parquet rows.
    ei["primary_key"] = ei.index.to_numpy()

    users = _user_descriptors(tables["users"])
    events = _event_descriptors(tables["events"], cluster_cols=True)

    stitched = ei.merge(users, on="user_id", how="left")
    stitched = stitched.merge(events, on="event_id", how="left")
    stitched["split"] = _split_from_date(stitched["timestamp"], valid_ts, test_ts)
    return stitched


# ---------------------------------------------------------------------------
# User-rooted parquets
# ---------------------------------------------------------------------------


def _build_interest_history_index(db) -> dict[int, pd.DataFrame]:
    """Return {user_id: DataFrame sorted by timestamp} for event_interest.

    Each per-user frame carries lightweight event descriptors and the two
    label columns folded in (the labels are historical facts once the row is
    strictly before ``as_of``, so they are legal features). ``user_id`` is
    dropped from the payload — it is redundant with the outer row's user.
    """
    tables = {name: table.df for name, table in db.table_dict.items()}
    ei = tables["event_interest"].rename(
        columns={"user": "user_id", "event": "event_id"}
    )
    events_light = _event_descriptors(tables["events"], cluster_cols=False)
    df = ei.merge(events_light, on="event_id", how="left")
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    payload_cols = [c for c in df.columns if c != "user_id"]
    return {
        int(uid): g[payload_cols].reset_index(drop=True)
        for uid, g in df.groupby("user_id", sort=False)
    }


def _build_attendance_history_index(db) -> dict[int, pd.DataFrame]:
    """Return {user_id: DataFrame sorted by start_time} for event_attendees.

    ``user_id`` is dropped from the payload for the same reason as
    :func:`_build_interest_history_index`.
    """
    tables = {name: table.df for name, table in db.table_dict.items()}
    ea = tables["event_attendees"].rename(columns={"event": "event_id"})
    ea = ea.drop(columns=[c for c in ("Unnamed: 0",) if c in ea.columns])
    ea = ea.dropna(subset=["user_id"])
    ea["user_id"] = ea["user_id"].astype("int64")
    events_light = _event_descriptors(tables["events"], cluster_cols=False)
    df = ea.merge(events_light, on="event_id", how="left")
    df = df.sort_values(["user_id", "start_time"]).reset_index(drop=True)
    payload_cols = [c for c in df.columns if c != "user_id"]
    return {
        int(uid): g[payload_cols].reset_index(drop=True)
        for uid, g in df.groupby("user_id", sort=False)
    }


def _build_friends_snapshot_index(db) -> dict[int, list[dict]]:
    """Return {user_id: [friend_record, ...]} bounded to FRIENDS_SNAPSHOT_LENGTH.

    Each friend record includes the friend's user descriptors so the schema
    can bind them without an in-query re-join.
    """
    tables = {name: table.df for name, table in db.table_dict.items()}
    uf_raw = tables["user_friends"]
    uf = uf_raw.drop(
        columns=[c for c in ("Unnamed: 0",) if c in uf_raw.columns]
    )
    uf = uf.dropna(subset=["friend"])
    uf["friend"] = uf["friend"].astype("int64")

    friend_descs = tables["users"].rename(columns={"user_id": "friend"})
    friend_descs = _prefix(friend_descs, "friend", keep={"friend"})
    joined = uf.merge(friend_descs, on="friend", how="left")
    joined = joined.sort_values(["user", "friend"]).reset_index(drop=True)

    payload_cols = [c for c in joined.columns if c != "user"]
    out: dict[int, list[dict]] = {}
    for uid, g in joined.groupby("user", sort=False):
        out[int(uid)] = g.head(FRIENDS_SNAPSHOT_LENGTH)[payload_cols].to_dict(
            orient="records"
        )
    return out


def _gather_recent(
    index: dict[int, pd.DataFrame],
    user_id: int,
    time_col: str,
    as_of: pd.Timestamp,
    length: int,
) -> list[dict]:
    sub = index.get(user_id)
    if sub is None or sub.empty:
        return []
    if as_of is None or pd.isna(as_of):
        window = sub
    else:
        window = sub[sub[time_col] < as_of]
    if window.empty:
        return []
    return window.tail(length).to_dict(orient="records")


def build_user_rooted_for_task(
    db,
    task_name: str,
    interest_len: int = INTEREST_HISTORY_LENGTH,
    attendance_len: int = ATTENDANCE_HISTORY_LENGTH,
) -> pd.DataFrame:
    """One row per ``(user, timestamp)`` from the RelBench task train / val /
    test tables, with the user's most recent context nested under
    ``recent_interests``, ``recent_attendance``, and ``friends``.

    Test-split rows have their label set to ``None`` (RelBench withholds
    test labels) but are still emitted so the trained model can produce a
    submission-ready prediction file.
    """
    spec: TaskSpec = TASKS[task_name]
    task = get_task("rel-event", task_name, download=True)

    tables = {name: table.df for name, table in db.table_dict.items()}
    users = _user_descriptors(tables["users"])

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

    # Normalize the entity / time column names so the rest of the code is
    # task-agnostic. RelBench uses ``user`` for the forecasting tasks and
    # ``user_id`` for the autocomplete ``users-birthyear`` task; ``timestamp``
    # for forecasting and ``joinedAt`` for the birthyear autocomplete.
    if task.entity_col != "user_id":
        all_rows = all_rows.rename(columns={task.entity_col: "user_id"})
    if task.time_col != "as_of":
        all_rows = all_rows.rename(columns={task.time_col: "as_of"})

    if spec.label_column != spec.target_path:
        all_rows = all_rows.rename(columns={spec.label_column: spec.target_path})

    # Attach user descriptors. For ``users-birthyear`` the raw label lives on
    # the users table itself, so drop the descriptor version to keep the
    # parquet leakage-clean without the caller having to remember.
    user_descs = users
    if spec.name == "users-birthyear" and "user_birthyear" in user_descs.columns:
        user_descs = user_descs.drop(columns=["user_birthyear"])
    all_rows = all_rows.merge(user_descs, on="user_id", how="left")

    print("  indexing event_interest history ...")
    interest_index = _build_interest_history_index(db)
    print("  indexing event_attendees history ...")
    attendance_index = _build_attendance_history_index(db)
    print("  indexing user_friends snapshot ...")
    friends_index = _build_friends_snapshot_index(db)

    print(f"  gathering context for {len(all_rows):,} observations ...")
    recent_interests: list[list[dict]] = []
    recent_attendance: list[list[dict]] = []
    friends: list[list[dict]] = []
    for uid, as_of in zip(all_rows["user_id"], all_rows["as_of"]):
        uid_int = int(uid)
        recent_interests.append(
            _gather_recent(interest_index, uid_int, "timestamp", as_of, interest_len)
        )
        recent_attendance.append(
            _gather_recent(attendance_index, uid_int, "start_time", as_of, attendance_len)
        )
        friends.append(friends_index.get(uid_int, []))
    all_rows["recent_interests"] = recent_interests
    all_rows["recent_attendance"] = recent_attendance
    all_rows["friends"] = friends

    return all_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def _summarize_history(df: pd.DataFrame, col: str) -> str:
    if col not in df.columns:
        return f"{col}=<absent>"
    lengths = df[col].map(len).to_numpy()
    if lengths.size == 0:
        return f"{col} avg=0.0 max=0"
    return f"{col} avg={lengths.mean():.1f} max={int(lengths.max())}"


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
        choices=[*TASKS, "event-interest-rooted", "all"],
        default="all",
        help="Build one specific task's parquet, the shared event-interest-"
             "rooted parquet, or every parquet (default).",
    )
    parser.add_argument(
        "--interest-history-length",
        type=int,
        default=INTEREST_HISTORY_LENGTH,
        help="Maximum number of past event_interest rows to nest under "
             "recent_interests for each user-rooted observation.",
    )
    parser.add_argument(
        "--attendance-history-length",
        type=int,
        default=ATTENDANCE_HISTORY_LENGTH,
        help="Maximum number of past event_attendees rows to nest under "
             "recent_attendance for each user-rooted observation.",
    )
    args = parser.parse_args()

    dataset = get_dataset("rel-event")
    db = dataset.get_db()
    valid_ts = pd.Timestamp(dataset.val_timestamp)
    test_ts = pd.Timestamp(dataset.test_timestamp)

    event_interest_tasks = [
        n for n, s in TASKS.items() if s.root_shape == "event_interest"
    ]
    user_tasks = [n for n, s in TASKS.items() if s.root_shape == "user"]

    build_ei_rooted = args.task in ("event-interest-rooted", "all") or (
        args.task in TASKS and TASKS[args.task].root_shape == "event_interest"
    )
    if build_ei_rooted:
        ei_df = build_event_interest_rooted(db, valid_ts, test_ts)
        ei_path = args.output_dir / EVENT_INTEREST_ROOTED_STEM
        _write(ei_df, ei_path)
        counts = ei_df["split"].value_counts().to_dict()
        print(
            f"[event-interest-rooted] {len(ei_df):,} rows -> {ei_path}  "
            f"splits={counts}  (serves: {event_interest_tasks})"
        )

    if args.task == "all":
        chosen_user_tasks = user_tasks
    elif args.task in TASKS and TASKS[args.task].root_shape == "user":
        chosen_user_tasks = [args.task]
    else:
        chosen_user_tasks = []

    for name in chosen_user_tasks:
        print(f"\n[{name}] building user-rooted parquet ...")
        df = build_user_rooted_for_task(
            db,
            name,
            interest_len=args.interest_history_length,
            attendance_len=args.attendance_history_length,
        )
        path = args.output_dir / TASKS[name].parquet_stem
        _write(df, path)
        counts = df["split"].value_counts().to_dict()
        print(
            f"[{name}] {len(df):,} rows -> {path}  splits={counts}  "
            f"{_summarize_history(df, 'recent_interests')}  "
            f"{_summarize_history(df, 'recent_attendance')}  "
            f"{_summarize_history(df, 'friends')}"
        )


if __name__ == "__main__":
    main()