"""Phase 2 - per-window END-TIME artifact (enables temporal CV).

The frozen artifacts persist `job_ids` (so GroupKFold/LeaveOneGroupOut are ready)
but no per-window timestamp, and `X_train` rows are ordered by (slurm_id,
timestamp) -- job-blocked, not globally time-sorted. `TimeSeriesSplit(gap=...)`
and the custom `PurgedGroupKFold` need a time key, so we materialize one here.

This does NOT change any feature, the window size, the labels, or `X`. It re-reads
only `[slurm_id, timestamp, end_date]` (with the same gpu_node=1 + CANCELLED-excluded
Arrow filters, the same sort, and the same 80/20 job split), windows the timestamp
column the SAME way, and takes each window's LAST snapshot time as the window
end-time. Output: `artifacts/win_time_{train,test}.npy` (int64 ns), aligned 1:1
with the rows of `X_{train,test}`.

Alignment is *proven*, not assumed: we rebuild the job-id-per-window sequence and
assert it equals the saved `job_ids_{train,test}.npy` element-for-element.

    python -m cv_strategies.timekey
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
# reuse the EXACT session-1 split logic so train_set is identical
from benchmark.data_pipeline import train_job_ids


def _load_scaffold() -> pd.DataFrame:
    """Only slurm_id / timestamp / end_date, GPU nodes, CANCELLED excluded -- all
    at the Arrow level so the 8 metric columns and `state` never enter pandas."""
    t0 = time.perf_counter()
    dataset = pq.ParquetDataset(
        str(C.DATASET_PATH),
        filters=[("gpu_node", "=", 1), ("state", "!=", "CANCELLED")],
    )
    df = dataset.read(columns=["slurm_id", "timestamp", "end_date"]).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    print(f"  loaded {len(df):,} snapshots / {df['slurm_id'].nunique():,} jobs "
          f"in {time.perf_counter() - t0:.1f}s")
    return df


def _window_times(df: pd.DataFrame, train_set: set):
    """Replicate build_windows' routing exactly, emitting window END-times and a
    parallel job-id sequence per split (for the alignment assertion)."""
    df = df.sort_values(["slurm_id", "timestamp"], kind="stable")
    buckets = {s: {"t": [], "j": []} for s in ("train", "test")}

    for slurm_id, job in df.groupby("slurm_id", sort=False):
        n = len(job)
        if n < C.WINDOW_SIZE:
            continue
        ts = job["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        win = np.lib.stride_tricks.sliding_window_view(ts, C.WINDOW_SIZE)
        end_times = win[:, -1]                      # last snapshot of each window
        split = "train" if slurm_id in train_set else "test"
        buckets[split]["t"].append(end_times)
        buckets[split]["j"].append(np.full(end_times.shape[0], slurm_id, dtype=np.int64))

    out = {}
    for split, b in buckets.items():
        out[split] = {
            "time": np.concatenate(b["t"]).astype(np.int64),
            "job": np.concatenate(b["j"]).astype(np.int64),
        }
    return out


def build() -> None:
    C.ensure_dirs()
    print("=" * 74)
    print("BUILD WINDOW-TIME ARTIFACT")
    print("=" * 74)
    df = _load_scaffold()
    train_set = train_job_ids(df)                    # identical to session 1
    win = _window_times(df, train_set)

    for split in ("train", "test"):
        t = win[split]["time"]
        j = win[split]["job"]
        # --- prove 1:1 alignment against the frozen job_ids artifact ---------- #
        saved_jobs = np.load(C.frozen_artifact_path(f"job_ids_{split}"))
        assert len(t) == len(saved_jobs), (
            f"{split}: {len(t)} window-times vs {len(saved_jobs)} saved rows")
        assert np.array_equal(j.astype(saved_jobs.dtype), saved_jobs), (
            f"{split}: rebuilt job-id sequence does not match job_ids_{split}.npy "
            "-> row order differs, win_time would be misaligned")
        np.save(C.WIN_TIME[f"win_time_{split}"], t)
        # sanity: end-times are non-decreasing within each contiguous job block
        first_job_mask = saved_jobs == saved_jobs[0]
        assert np.all(np.diff(t[first_job_mask]) >= 0), \
            f"{split}: first job's window end-times are not monotonic"
        dt = pd.to_datetime(t.min()), pd.to_datetime(t.max())
        print(f"  {split:<5} rows={len(t):>10,}  aligned OK  "
              f"span [{dt[0].date()} .. {dt[1].date()}]  -> "
              f"{C.WIN_TIME[f'win_time_{split}'].name}")

    print("=" * 74)
    print("  window-time artifacts written; 1:1 alignment with X_* verified.")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
