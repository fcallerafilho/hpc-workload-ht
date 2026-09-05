"""Data pipeline: parquet -> label -> temporal split -> sliding window -> .npy.

Run once; the artifacts are reused by every model in Pass 1 and Pass 2.

    python -m benchmark.data_pipeline            # skip if artifacts exist
    python -m benchmark.data_pipeline --force    # rebuild from scratch

Authoritative spec: TCC_Dados_ML.pdf. Where the PDF is silent on mechanics
(std ddof, majority tie-break) we follow dataset/ml_pipeline.ipynb:
  * per-snapshot label = failure job AND within 2 h before end_date
  * temporal split: first 80 % of jobs (sorted by end_date) -> train
  * per-slurm_id sliding window, size 40, stride 1
  * window label = majority (>= 20 positives -> 1)
  * aggregations min/max/mean/std, np.std ddof=0

Memory strategy (16 GB laptop): load only 14 of 100 columns via PyArrow with a
gpu_node=1 predicate; route each job to train/test inside a single groupby pass
(no df_train/df_test copies); build float32 windows; free the raw frame before
the final vstack. states -> int8 codes, job_ids -> int32.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import get_logger, human_bytes

log = get_logger()


def load_frame() -> pd.DataFrame:
    """Load the 14 needed columns for GPU nodes, excluding CANCELLED jobs."""
    columns = config.FEATURES + config.SCAFFOLD
    log.info("Loading parquet (14/100 cols, gpu_node=1 predicate)  from %s",
             config.DATASET_PATH)
    t0 = time.perf_counter()
    dataset = pq.ParquetDataset(str(config.DATASET_PATH), filters=[("gpu_node", "=", 1)])
    df = dataset.read(columns=columns).to_pandas()
    df = df[df["state"] != "CANCELLED"].reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    log.info("Loaded %s snapshots, %s jobs in %.1f s  (~%s in RAM)",
             f"{len(df):,}", f"{df['slurm_id'].nunique():,}", time.perf_counter() - t0,
             human_bytes(df.memory_usage(deep=True).sum()))
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot label: 1 iff failure job and snapshot in the 2 h pre-end window."""
    is_failure = df["state"].isin(config.FAILURE_STATES)
    time_to_end = df["end_date"] - df["timestamp"]
    in_window = (time_to_end <= pd.Timedelta(hours=config.PRE_FAILURE_HOURS)) & \
                (time_to_end >= pd.Timedelta(0))
    df["label"] = (is_failure & in_window).astype(np.int8)
    log.info("Snapshot labels: %s positive / %s total  (%.4f)",
             f"{int(df['label'].sum()):,}", f"{len(df):,}", df["label"].mean())
    return df


def train_job_ids(df: pd.DataFrame) -> set:
    """First 80 % of jobs sorted by end_date go to train (per-job temporal split)."""
    job_end = df.groupby("slurm_id")["end_date"].first().sort_values()
    n_train = int(len(job_end) * config.TRAIN_FRACTION)
    split_time = job_end.iloc[n_train]
    log.info("Temporal split at end_date %s : %s train jobs / %s test jobs",
             split_time, f"{n_train:,}", f"{len(job_end) - n_train:,}")
    return set(job_end.iloc[:n_train].index)


def _window_job(arr32: np.ndarray, labels: np.ndarray):
    """Aggregate one job's snapshots into (n_windows, 32) float32 + majority labels.

    arr32 is (n_snapshots, 8) float32. Reductions are done in float64 (jobs are
    small) to avoid float32 cancellation on the ~1e10 byte metrics, then cast
    back to float32.
    """
    arr = arr32.astype(np.float64, copy=False)
    win = np.lib.stride_tricks.sliding_window_view(arr, config.WINDOW_SIZE, axis=0)
    agg = np.concatenate(
        [win.min(-1), win.max(-1), win.mean(-1), win.std(-1)], axis=1
    ).astype(np.float32)
    lab_win = np.lib.stride_tricks.sliding_window_view(labels, config.WINDOW_SIZE)
    win_labels = (lab_win.sum(1) >= config.WINDOW_SIZE // 2).astype(np.int8)
    return agg, win_labels


def build_windows(df: pd.DataFrame, train_set: set):
    """Single groupby pass: route each job's windows to the train or test buckets.

    Returns two dicts (train, test) each with X, y, states, job_ids.
    """
    df = df.sort_values(["slurm_id", "timestamp"], kind="stable")
    feat = config.FEATURES

    buckets = {
        split: {"X": [], "y": [], "states": [], "job_ids": []}
        for split in ("train", "test")
    }
    kept = {"train": 0, "test": 0}
    dropped = 0

    for slurm_id, job in df.groupby("slurm_id", sort=False):
        n = len(job)
        if n < config.WINDOW_SIZE:
            dropped += 1
            continue
        split = "train" if slurm_id in train_set else "test"
        arr32 = job[feat].to_numpy(dtype=np.float32)
        labels = job["label"].to_numpy()
        agg, win_labels = _window_job(arr32, labels)
        n_win = agg.shape[0]

        state_code = config.STATE_CODES[job["state"].iloc[0]]
        b = buckets[split]
        b["X"].append(agg)
        b["y"].append(win_labels)
        b["states"].append(np.full(n_win, state_code, dtype=np.int8))
        b["job_ids"].append(np.full(n_win, slurm_id, dtype=np.int64))
        kept[split] += 1

    log.info("Windowed jobs kept: %s train / %s test  (dropped %s jobs < %d snapshots)",
             f"{kept['train']:,}", f"{kept['test']:,}", f"{dropped:,}", config.WINDOW_SIZE)
    return buckets


def _stack(bucket: dict, id_dtype) -> dict:
    X = np.vstack(bucket["X"])
    y = np.concatenate(bucket["y"])
    states = np.concatenate(bucket["states"])
    job_ids = np.concatenate(bucket["job_ids"]).astype(id_dtype, copy=False)
    # free the per-job lists as we go
    bucket["X"].clear(); bucket["y"].clear(); bucket["states"].clear(); bucket["job_ids"].clear()
    return {"X": X, "y": y, "states": states, "job_ids": job_ids}


def run(force: bool = False) -> None:
    if config.artifacts_exist() and not force:
        log.info("Artifacts already present in %s — skipping (use --force to rebuild).",
                 config.ARTIFACTS_DIR)
        return

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_frame()
    df = add_labels(df)
    train_set = train_job_ids(df)

    # Decide job_id dtype: int32 if slurm_ids fit, else int64.
    max_id = int(df["slurm_id"].max())
    id_dtype = np.int32 if max_id < 2**31 - 1 else np.int64
    log.info("max slurm_id = %s -> job_ids dtype %s", f"{max_id:,}", np.dtype(id_dtype).name)

    buckets = build_windows(df, train_set)
    del df
    gc.collect()  # free the ~2 GB raw frame before the vstack

    train = _stack(buckets["train"], id_dtype)
    test = _stack(buckets["test"], id_dtype)
    del buckets
    gc.collect()

    for split, d in (("train", train), ("test", test)):
        pos = int(d["y"].sum())
        log.info("X_%s: %s  |  positives: %s (%.4f)",
                 split, d["X"].shape, f"{pos:,}", d["y"].mean())

    # --- persist arrays --------------------------------------------------- #
    np.save(config.artifact_path("X_train"), train["X"])
    np.save(config.artifact_path("y_train"), train["y"])
    np.save(config.artifact_path("states_train"), train["states"])
    np.save(config.artifact_path("job_ids_train"), train["job_ids"])
    np.save(config.artifact_path("X_test"), test["X"])
    np.save(config.artifact_path("y_test"), test["y"])
    np.save(config.artifact_path("states_test"), test["states"])
    np.save(config.artifact_path("job_ids_test"), test["job_ids"])

    with open(config.artifact_path("feature_names"), "w", encoding="utf-8") as fh:
        json.dump(config.feature_names(), fh, indent=2)
    with open(config.artifact_path("state_names"), "w", encoding="utf-8") as fh:
        json.dump({str(c): s for c, s in config.CODE_TO_STATE.items()}, fh, indent=2)

    # --- scaler (fit on X_train only) ------------------------------------- #
    log.info("Fitting StandardScaler on X_train...")
    scaler = StandardScaler().fit(train["X"])
    joblib.dump(scaler, config.artifact_path("scaler"))

    total = sum(config.artifact_path(k).stat().st_size
                for k in config.ARTIFACTS if config.artifact_path(k).exists())
    log.info("Artifacts written to %s  (total %s)", config.ARTIFACTS_DIR, human_bytes(total))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build windowed .npy artifacts.")
    ap.add_argument("--force", action="store_true", help="rebuild even if artifacts exist")
    args = ap.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
