"""Temporal splits (P9): train on the past, test on the future.

Jobs are ordered by end time (ties broken by job_id).  The main split puts the first 80 % of the
jobs in training and the last 20 % in test.  The three blocks are consecutive and each holds about
one third of the target-(a) positives; each block has its own 80/20 split.  No cut separates two
jobs with the same end time, so every training end time precedes every test end time.
"""
import numpy as np
import pandas as pd


def _cut(t: np.ndarray, i: int) -> int:
    """Move cut position i forward until t[i - 1] < t[i] (no tie across the cut)."""
    while 0 < i < len(t) and t[i - 1] == t[i]:
        i += 1
    return i


def temporal_splits(df: pd.DataFrame, end_col: str = "end_time", test_frac: float = 0.2) -> dict:
    """{"main" | "block1" | "block2" | "block3": {"train": job_ids, "test": job_ids}}.

    `df` holds one population with all five states, and the columns job_id, target_a, end_col.
    """
    d = df.sort_values([end_col, "job_id"]).reset_index(drop=True)
    t, ids, n = d[end_col].values, d["job_id"].values, len(d)
    out = {}

    i = _cut(t, int(round((1 - test_frac) * n)))
    out["main"] = {"train": ids[:i], "test": ids[i:]}

    cum_pos = np.cumsum(d["target_a"].values == 1)
    total = cum_pos[-1]
    b1 = _cut(t, int(np.searchsorted(cum_pos, total / 3)) + 1)
    b2 = _cut(t, int(np.searchsorted(cum_pos, 2 * total / 3)) + 1)
    for k, (a, b) in enumerate([(0, b1), (b1, b2), (b2, n)], start=1):
        j = _cut(t[a:b], int(round((1 - test_frac) * (b - a))))
        out[f"block{k}"] = {"train": ids[a:a + j], "test": ids[a + j:b]}
    return out
