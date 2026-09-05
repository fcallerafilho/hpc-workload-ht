"""Phase 3 - cross-validation strategies for overlapping windowed data.

The session-1 spec used `StratifiedKFold(shuffle=True)`. That LEAKS here: the
sliding window has 39/40 overlap, so windows of the *same job* are near-duplicates;
shuffling scatters them across folds, and every validation fold contains rows that
are almost identical to rows it "trained" on -> CV-AP is optimistic (~0.6-0.7) while
the true temporal test-AP is ~0.06. The professor's fix is grouped / time-aware CV.

`make_splits(name, y, groups, times)` returns a concrete list of (train_idx,
test_idx) pairs (original positions), which is passed straight to
`GridSearchCV(cv=splits)` -- no groups-forwarding needed.

Strategies:
  stratified        StratifiedKFold(3, shuffle) .......... the leaky baseline, kept for contrast
  group             GroupKFold(3), groups=slurm_id ....... whole job in one fold (no overlap leak)
  stratified_group  StratifiedGroupKFold(3, shuffle) ..... group integrity + class balance
  tscv              TimeSeriesSplit(5, gap=39) by time ... train past -> validate future
  purged            PurgedGroupKFold(5) .................. time-blocked folds, jobs kept whole,
                                                           train jobs overlapping the val time
                                                           interval (+2 h embargo) are purged
  logo_weekly       LeaveOneGroupOut over ISO-week buckets (per-job LOGO = ~18 k folds, infeasible)

Run `python -m cv_strategies.cv` for the self-tests (leakage guarantees).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from . import config as C

# label-dependency horizon used as the purge embargo (labels look back 2 h)
EMBARGO_NS = int(C.PRE_FAILURE_HOURS * 3600 * 1_000_000_000)
GAP = C.WINDOW_SIZE - 1                      # 39 overlapping windows at a boundary

STRATEGIES = ["stratified", "group", "stratified_group", "tscv", "purged", "logo_weekly"]
NEEDS_TIMES = {"tscv", "purged", "logo_weekly"}
NEEDS_GROUPS = {"group", "stratified_group", "purged", "logo_weekly"}


# --------------------------------------------------------------------------- #
# Custom splitters (return original-position index arrays)
# --------------------------------------------------------------------------- #
def _tscv_by_time(times: np.ndarray, n_splits: int = 5, gap: int = GAP):
    """TimeSeriesSplit applied in time-sorted space; yields original positions."""
    order = np.argsort(times, kind="stable")
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    dummy = np.zeros(len(times), dtype=np.int8)
    for tr, te in tss.split(dummy):
        yield order[tr], order[te]


def _job_bounds(groups: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """Per-job [start, end] window end-times and row indices."""
    g = pd.DataFrame({"job": groups, "t": times})
    agg = g.groupby("job")["t"].agg(["min", "max"])
    agg["idx"] = g.groupby("job").indices           # dict-like -> array per job
    return agg


def _purged_group_kfold(groups: np.ndarray, times: np.ndarray,
                        n_splits: int = 5, embargo_ns: int = EMBARGO_NS):
    """Time-blocked K-fold at the JOB level (no job split), purging train jobs whose
    time interval overlaps the validation interval expanded by `embargo_ns`."""
    b = _job_bounds(groups, times)
    b = b.sort_values("max", kind="stable")          # order jobs temporally by end
    jobs = b.index.to_numpy()
    starts = b["min"].to_numpy()
    ends = b["max"].to_numpy()
    idx_of_job = b["idx"].to_dict()

    for block in np.array_split(np.arange(len(jobs)), n_splits):
        if len(block) == 0:
            continue
        val_start = starts[block].min()
        val_end = ends[block].max()
        lo, hi = val_start - embargo_ns, val_end + embargo_ns

        val_jobs = set(jobs[block].tolist())
        val_idx = np.concatenate([idx_of_job[j] for j in jobs[block]])

        train_jobs = []
        for k, j in enumerate(jobs):
            if j in val_jobs:
                continue
            if ends[k] < lo or starts[k] > hi:       # entirely outside embargoed val span
                train_jobs.append(j)
        train_idx = np.concatenate([idx_of_job[j] for j in train_jobs])
        yield np.sort(train_idx), np.sort(val_idx)


def _weekly_logo(groups: np.ndarray, times: np.ndarray):
    """LeaveOneGroupOut over ISO-week buckets; each job assigned to ONE week (its
    end-time's week) so jobs are never split across folds."""
    b = _job_bounds(groups, times)
    wk = pd.to_datetime(b["max"]).dt.isocalendar()
    job_week = (wk["year"].astype(int) * 100 + wk["week"].astype(int))
    week_of_job = job_week.to_dict()
    win_week = pd.Series(groups).map(week_of_job).to_numpy()

    logo = LeaveOneGroupOut()
    dummy = np.zeros(len(groups), dtype=np.int8)
    for tr, te in logo.split(dummy, groups=win_week):
        yield tr, te


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def make_splits(name: str, *, y=None, groups=None, times=None,
                n_splits: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a concrete [(train_idx, test_idx), ...] list for strategy `name`."""
    if name in NEEDS_TIMES and times is None:
        raise ValueError(f"strategy '{name}' needs `times` (run timekey.py first)")
    if name in NEEDS_GROUPS and groups is None:
        raise ValueError(f"strategy '{name}' needs `groups` (job_ids)")

    if name == "stratified":
        k = n_splits or 3
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=C.SEED)
        return list(cv.split(np.zeros(len(y)), y))
    if name == "group":
        k = n_splits or 3
        cv = GroupKFold(n_splits=k)
        return list(cv.split(np.zeros(len(groups)), y, groups))
    if name == "stratified_group":
        k = n_splits or 3
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=C.SEED)
        return list(cv.split(np.zeros(len(groups)), y, groups))
    if name == "tscv":
        return list(_tscv_by_time(times, n_splits=n_splits or 5))
    if name == "purged":
        return list(_purged_group_kfold(groups, times, n_splits=n_splits or 5))
    if name == "logo_weekly":
        return list(_weekly_logo(groups, times))
    raise ValueError(f"unknown CV strategy '{name}' (choices: {STRATEGIES})")


# --------------------------------------------------------------------------- #
# Self-tests: verify the leakage guarantees on the TRAIN split
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    a = C.load_frozen(mmap=True)
    y = np.asarray(a["y_train"])
    groups = np.asarray(a["job_ids_train"])
    times, _ = C.load_win_time()
    print("=" * 74)
    print("CV SELF-TESTS (train split: %d rows, %d jobs)" % (len(y), len(np.unique(groups))))
    print("=" * 74)

    for name in STRATEGIES:
        splits = make_splits(name, y=y, groups=groups, times=times)
        n = len(splits)
        # no train/val row index overlap, ever
        for tr, te in splits:
            assert len(np.intersect1d(tr, te)) == 0, f"{name}: train/val index overlap"

        msg = f"  {name:<17} folds={n:<3}"

        if name in ("group", "stratified_group", "purged", "logo_weekly"):
            # group integrity: no job appears on both sides of any fold
            for tr, te in splits:
                assert set(groups[tr]).isdisjoint(set(groups[te])), \
                    f"{name}: a job is in both train and val"
            msg += "  job-integrity OK"

        if name == "tscv":
            # every training end-time precedes every validation end-time (+gap)
            for tr, te in splits:
                assert times[tr].max() <= times[te].min(), f"{name}: train not before val"
            msg += "  train<val OK"

        if name == "purged":
            # no train window's end-time lies inside the embargoed validation span
            for tr, te in splits:
                lo, hi = times[te].min() - EMBARGO_NS, times[te].max() + EMBARGO_NS
                inside = ((times[tr] >= lo) & (times[tr] <= hi)).sum()
                assert inside == 0, f"{name}: {inside} train rows inside embargoed val span"
            msg += "  purge OK"

        # class balance report (val positive fraction range across folds)
        val_fracs = [float((y[te] == 1).mean()) for _, te in splits]
        msg += f"  val_pos_frac[min={min(val_fracs):.4f} max={max(val_fracs):.4f}]"
        print(msg)

    print("=" * 74)
    print("  all CV self-tests passed.")


if __name__ == "__main__":
    _selftest()
