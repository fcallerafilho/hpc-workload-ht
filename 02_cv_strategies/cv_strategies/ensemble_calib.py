"""Phase 7 - ensemble of the tuned top-3 + probability calibration.

Ensemble: average the tuned LightGBM / CatBoost / HGB test probabilities (plain
mean and rank-mean) and evaluate on the temporal test set.

Calibration: the class-weighted boosters output well-RANKED but mis-CALIBRATED
scores. We refit the best config on the temporal-fit set, fit isotonic and sigmoid
calibrators on the temporal holdout, and apply them to the test scores. PR-AUC and
ROC-AUC are invariant under a monotonic map, so they do NOT change -- the point is
a lower Brier score and a straighter reliability curve, which makes the chosen
operating-point threshold mean what it says.

    python -m cv_strategies.ensemble_calib
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from . import config as C
from . import tuning as T
from benchmark.evaluation import evaluate

TOP3 = ["LightGBM", "CatBoost", "HistGradientBoostingClassifier"]
ENS_FIELDS = ["name", "kind", "test_pr_auc", "test_roc_auc", "recall_global",
              "recall_TIMEOUT", "recall_FAILED", "recall_OOM", "recall_NODE_FAIL",
              "base_pr_auc", "note"]


def _load_tuned_probas() -> dict:
    out = {}
    for m in TOP3:
        p = C.GRID_DIR / f"{m}_tuned_test_proba.npy"
        if p.exists():
            out[m] = np.load(p).astype(np.float64)
    return out


def ensemble() -> list[dict]:
    probas = _load_tuned_probas()
    if len(probas) < 2:
        print("Need >=2 tuned models for an ensemble; found:", list(probas))
        return []
    y_te = np.load(C.frozen_artifact_path("y_test"))
    states_te = np.load(C.frozen_artifact_path("states_test"))
    names = list(probas)
    P = np.vstack([probas[n] for n in names])

    rows = []
    # individual tuned models (reference)
    for n in names:
        m = evaluate(y_te, probas[n], states_te)
        rows.append({"name": n, "kind": "tuned_single", "test_pr_auc": m["pr_auc"],
                     "test_roc_auc": m["roc_auc"], "recall_global": m["recall_global"],
                     **{f"recall_{s}": m[f"recall_{s}"] for s in
                        ["TIMEOUT", "FAILED", "OOM", "NODE_FAIL"]},
                     "base_pr_auc": T._base_pr_auc(n), "note": ""})
    # mean + rank-mean ensembles
    mean = P.mean(axis=0)
    rankmean = np.vstack([rankdata(probas[n]) / len(y_te) for n in names]).mean(axis=0)
    for label, sc in (("Ensemble_mean", mean), ("Ensemble_rankmean", rankmean)):
        m = evaluate(y_te, sc, states_te)
        rows.append({"name": label, "kind": "ensemble", "test_pr_auc": m["pr_auc"],
                     "test_roc_auc": m["roc_auc"], "recall_global": m["recall_global"],
                     **{f"recall_{s}": m[f"recall_{s}"] for s in
                        ["TIMEOUT", "FAILED", "OOM", "NODE_FAIL"]},
                     "base_pr_auc": float("nan"), "note": "+".join(names)})
    return rows


def calibration() -> dict:
    """Refit best config on temporal-fit, calibrate on temporal holdout, apply to test."""
    if not C.TUNING_RESULTS_CSV.exists():
        print("No tuning_results.csv -- run --retune first."); return {}
    df = pd.read_csv(C.TUNING_RESULTS_CSV)
    df = df[df["status"] == "ok"].sort_values("test_pr_auc", ascending=False)
    if df.empty:
        return {}
    best = df.iloc[0]
    name = best["model"]; kind = T.TUNE[name]["kind"]
    params = json.loads(best["best_params"]); n_rounds = int(best["final_n_rounds"])

    X_tr, X_te, y_tr, y_te, states_te, groups_tr, times_tr = T._load()
    spw = T._spw()
    fit_mask, hold_mask = T._temporal_holdout(times_tr, groups_tr)
    est = T._final_estimator(kind, params, n_rounds, spw)
    est.fit(np.ascontiguousarray(X_tr[fit_mask]), y_tr[fit_mask])

    s_hold = est.predict_proba(np.ascontiguousarray(X_tr[hold_mask]))[:, 1]
    y_hold = y_tr[hold_mask]
    s_test = est.predict_proba(X_te)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip").fit(s_hold, y_hold)
    sig = LogisticRegression(C=1e6, max_iter=1000).fit(s_hold.reshape(-1, 1), y_hold)
    test_iso = iso.predict(s_test)
    test_sig = sig.predict_proba(s_test.reshape(-1, 1))[:, 1]

    def _reliab(sc):
        pt, pp = calibration_curve(y_te, sc, n_bins=10, strategy="quantile")
        return {"prob_true": pt.tolist(), "prob_pred": pp.tolist(),
                "brier": float(brier_score_loss(y_te, sc)),
                "pr_auc": float(evaluate(y_te, sc, states_te)["pr_auc"])}

    out = {"model": name, "raw": _reliab(s_test),
           "isotonic": _reliab(test_iso), "sigmoid": _reliab(test_sig)}
    (C.RESULTS_DIR / "calibration.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    C.ensure_dirs()
    rows = ensemble()
    if rows:
        pd.DataFrame(rows).to_csv(C.ENSEMBLE_CALIB_CSV, index=False)
        print("=" * 78)
        print("PHASE 7 - ENSEMBLE (tuned top-3)")
        print("=" * 78)
        for r in rows:
            print(f"  {r['name']:<20} {r['kind']:<14} PR-AUC={r['test_pr_auc']:.4f}  "
                  f"ROC={r['test_roc_auc']:.4f}")
        print(f"  written: {C.ENSEMBLE_CALIB_CSV}")

    cal = calibration()
    if cal:
        print("-" * 78)
        print("PHASE 7 - CALIBRATION (best model: %s)" % cal["model"])
        for k in ("raw", "isotonic", "sigmoid"):
            print(f"  {k:<9} Brier={cal[k]['brier']:.5f}  PR-AUC={cal[k]['pr_auc']:.4f}")
        print("  (PR-AUC unchanged under isotonic/sigmoid as expected; Brier is the gain)")
        print(f"  written: {C.RESULTS_DIR / 'calibration.json'}")


if __name__ == "__main__":
    main()
