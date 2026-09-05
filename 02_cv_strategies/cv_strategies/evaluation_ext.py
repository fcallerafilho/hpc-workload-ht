"""Metric interpretation helpers + Phase 6 operating-point selection.

Extends (does not replace) session-1 `benchmark.evaluation`. Adds the baseline-
relative reading the results were missing:
  * lift        = PR-AUC / baseline (baseline = TEST prevalence ~0.0403)
  * verdict     = a coarse band so "is this good?" has an answer
  * PR / ROC curve arrays for plotting with baseline reference lines
  * threshold selection for a deployable operating point (target recall, best F-beta)

Phase 6 (`python -m cv_strategies.evaluation_ext`) loads the best tuned model's
test scores and writes `results/operating_points.csv`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config as C


# --------------------------------------------------------------------------- #
# interpretation
# --------------------------------------------------------------------------- #
def baseline() -> float:
    """PR-AUC random baseline = TEST positive prevalence (from imbalance.json,
    falling back to a direct compute)."""
    if C.IMBALANCE_JSON.exists():
        return float(json.loads(C.IMBALANCE_JSON.read_text())["pr_auc_baseline"])
    y = np.load(C.frozen_artifact_path("y_test"))
    return float((y == 1).mean())


def lift(pr_auc: float, base: float | None = None) -> float:
    return pr_auc / (base if base is not None else baseline())


def pr_verdict(pr_auc: float, base: float | None = None) -> str:
    """Coarse, honest band for an extreme-imbalance PR-AUC, expressed as x-lift
    over the random baseline (heuristic, for reader orientation only)."""
    x = lift(pr_auc, base)
    if x < 1.2:
        return "~random (<1.2x baseline)"
    if x < 2.0:
        return "weak (1.2-2x)"
    if x < 4.0:
        return "moderate (2-4x)"
    if x < 8.0:
        return "strong (4-8x)"
    return "excellent (>8x)"


def roc_verdict(roc: float) -> str:
    if roc < 0.6:
        return "poor (~random)"
    if roc < 0.7:
        return "weak"
    if roc < 0.8:
        return "fair"
    if roc < 0.9:
        return "good"
    return "excellent"


# --------------------------------------------------------------------------- #
# curves
# --------------------------------------------------------------------------- #
def pr_points(y_true, y_score) -> dict:
    p, r, t = precision_recall_curve(y_true, y_score)
    return {"precision": p, "recall": r, "thresholds": t,
            "ap": float(average_precision_score(y_true, y_score))}


def roc_points(y_true, y_score) -> dict:
    fpr, tpr, t = roc_curve(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "thresholds": t,
            "auc": float(roc_auc_score(y_true, y_score))}


# --------------------------------------------------------------------------- #
# threshold selection
# --------------------------------------------------------------------------- #
def threshold_for_target_recall(y_true, y_score, target: float = 0.80) -> dict:
    """Highest-precision threshold that still achieves >= `target` recall."""
    p, r, t = precision_recall_curve(y_true, y_score)
    # p,r have len = len(t)+1; drop the last (recall=0, threshold=+inf) sentinel
    p, r = p[:-1], r[:-1]
    ok = r >= target
    if not ok.any():
        idx = int(np.argmax(r))                     # cannot reach target -> max recall
    else:
        idx = int(np.argmax(np.where(ok, p, -1)))   # best precision among ok
    return {"threshold": float(t[idx]), "precision": float(p[idx]), "recall": float(r[idx])}


def threshold_best_fbeta(y_true, y_score, beta: float = 2.0) -> dict:
    """Threshold maximizing F-beta (beta>1 favors recall -- right for proactive
    node draining, where a missed failure costs more than a false drain)."""
    p, r, t = precision_recall_curve(y_true, y_score)
    p, r = p[:-1], r[:-1]
    b2 = beta * beta
    denom = (b2 * p) + r
    fbeta = np.divide((1 + b2) * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    idx = int(np.argmax(fbeta))
    return {"threshold": float(t[idx]), "precision": float(p[idx]),
            "recall": float(r[idx]), "fbeta": float(fbeta[idx]), "beta": beta}


def per_state_recall(y_true, y_score, states, threshold: float) -> dict:
    """Fraction of each failure state's positive windows flagged at `threshold`."""
    y_true = np.asarray(y_true); states = np.asarray(states)
    pred = np.asarray(y_score) >= threshold
    out = {}
    for raw in C.REPORT_FAILURE_STATES:
        disp = C.STATE_DISPLAY[raw]
        m = (states == C.STATE_CODES[raw]) & (y_true == 1)
        out[disp] = float((pred & m).sum() / m.sum()) if m.sum() else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Phase 6 runner
# --------------------------------------------------------------------------- #
def _best_tuned():
    """(model_name, test_proba) for the highest-PR-AUC tuned model, else None."""
    if not C.TUNING_RESULTS_CSV.exists():
        return None
    df = pd.read_csv(C.TUNING_RESULTS_CSV)
    df = df[df["status"] == "ok"].sort_values("test_pr_auc", ascending=False)
    if df.empty:
        return None
    name = df.iloc[0]["model"]
    p = C.GRID_DIR / f"{name}_tuned_test_proba.npy"
    return (name, np.load(p)) if p.exists() else None


def operating_points() -> None:
    C.ensure_dirs()
    picked = _best_tuned()
    if picked is None:
        print("No tuned model found -- run `--retune` first.")
        return
    name, proba = picked
    y_te = np.load(C.frozen_artifact_path("y_test"))
    states_te = np.load(C.frozen_artifact_path("states_test"))

    ops = {
        "default@0.5": {"threshold": 0.5},
        "target_recall@0.80": threshold_for_target_recall(y_te, proba, 0.80),
        "best_F2": threshold_best_fbeta(y_te, proba, 2.0),
    }
    rows = []
    for op_name, op in ops.items():
        thr = op["threshold"]
        pred = proba >= thr
        tp = int((pred & (y_te == 1)).sum())
        fp = int((pred & (y_te == 0)).sum())
        fn = int((~pred & (y_te == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) else 0.0
        row = {"model": name, "operating_point": op_name, "threshold": thr,
               "precision": prec, "recall": rec, "f2": f2,
               "flagged": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn}
        row.update({f"recall_{k}": v for k, v in per_state_recall(y_te, proba, states_te, thr).items()})
        rows.append(row)

    pd.DataFrame(rows).to_csv(C.OPERATING_POINTS_CSV, index=False)
    print("=" * 78)
    print(f"PHASE 6 - OPERATING POINTS  (model: {name})")
    print("=" * 78)
    for r in rows:
        print(f"  {r['operating_point']:<20} thr={r['threshold']:.3f}  "
              f"precision={r['precision']:.3f}  recall={r['recall']:.3f}  F2={r['f2']:.3f}  "
              f"flagged={r['flagged']:,}")
    print(f"  written: {C.OPERATING_POINTS_CSV}")


if __name__ == "__main__":
    operating_points()
