"""Single evaluation function shared by Pass 1 and Pass 2.

Given (y_true, y_score, state_codes) it returns every metric the benchmark
reports: PR-AUC (primary), ROC-AUC, global recall@0.5, per-state recall@0.5
for the four failure states, and per-state mean predicted probability
(diagnostic, including COMPLETED).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from . import config


def evaluate(y_true: np.ndarray, y_score: np.ndarray, state_codes: np.ndarray,
             threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    state_codes = np.asarray(state_codes)
    y_pred = (y_score >= threshold).astype(np.int8)

    out: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_global": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    # Per-state recall@0.5 (failure states only -- COMPLETED has no positives).
    for raw in config.REPORT_FAILURE_STATES:
        disp = config.STATE_DISPLAY[raw]
        mask = state_codes == config.STATE_CODES[raw]
        n_pos = int((y_true[mask] == 1).sum()) if mask.any() else 0
        if n_pos > 0:
            hits = int(((y_score[mask] >= threshold) & (y_true[mask] == 1)).sum())
            out[f"recall_{disp}"] = hits / n_pos
        else:
            out[f"recall_{disp}"] = float("nan")

    # Per-state mean predicted probability (diagnostic ranking signal).
    for raw in config.REPORT_ALL_STATES:
        disp = config.STATE_DISPLAY[raw]
        mask = state_codes == config.STATE_CODES[raw]
        out[f"mean_proba_{disp}"] = float(y_score[mask].mean()) if mask.any() else float("nan")

    return out


# Column order for results.csv (metric block; run scripts prepend the id block).
METRIC_COLUMNS = (
    ["pr_auc", "roc_auc", "recall_global"]
    + [f"recall_{config.STATE_DISPLAY[s]}" for s in config.REPORT_FAILURE_STATES]
    + [f"mean_proba_{config.STATE_DISPLAY[s]}" for s in config.REPORT_ALL_STATES]
)
