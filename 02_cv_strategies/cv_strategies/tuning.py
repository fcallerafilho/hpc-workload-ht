"""Phases 4 & 5 - CV diagnosis and honest re-tune.

Phase 4 (`--diagnostic`): fit a small LightGBM grid under EACH CV strategy and
record best CV-AP vs the true test PR-AUC. Expected signature: the leaky
`stratified` CV-AP is hugely optimistic while grouped/temporal CV-AP tracks the
test score -> the concrete evidence for the professor.

Phase 5 (`--retune`): re-run trimmed grids for LightGBM / CatBoost / HGB under an
honest CV (default `purged`), then finalize the number of boosting rounds with
temporal early stopping on a held-out tail of the training set (last 15% of jobs
by time -- never the test set). Evaluate the best estimator on the temporal test
set. Success = tuned test PR-AUC >= the session-1 base.

    python -m cv_strategies.tuning --diagnostic
    python -m cv_strategies.tuning --retune --cv purged
    python -m cv_strategies.tuning --retune --cv group --models LightGBM,CatBoost
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV

from . import config as C
from . import cv as CV
from benchmark.evaluation import evaluate          # reuse session-1 metric fn
from benchmark.utils import append_row, PeakMemorySampler

# --------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------- #
def _spw() -> float:
    y = np.load(C.frozen_artifact_path("y_train"))
    return float((y == 0).sum() / (y == 1).sum())


def _lgbm(**kw):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_jobs=-1, random_state=C.SEED, verbose=-1, **kw)


def _catb(**kw):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(auto_class_weights="Balanced", verbose=0,
                              random_seed=C.SEED, thread_count=-1,
                              allow_writing_files=False, **kw)


DIAG_GRID = {"num_leaves": [31], "min_child_samples": [20, 100], "learning_rate": [0.1]}
DIAG_N_ESTIMATORS = 200

MAX_ROUNDS = 600        # early-stopping ceiling
ES_PATIENCE = 60        # rounds of no temporal-holdout improvement before stopping
HGB_STAGES = [25, 50, 75, 100, 150, 200, 300, 400]   # warm-start staged max_iter grid

# Structural grids for the honest re-tune. The number of boosting rounds is NOT a
# grid axis: it is finalized per candidate by early stopping on a TEMPORAL holdout
# (the latest 15% of train jobs). We proved that a same-period monitor overfits the
# future -- only the temporal monitor selects rounds that generalize to the test.
TUNE = {
    "LightGBM": {
        "kind": "lgbm",
        "grid": {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1],
                 "min_child_samples": [20, 100]},
    },
    "CatBoost": {
        "kind": "catboost",
        "grid": {"depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
    },
    "HistGradientBoostingClassifier": {
        "kind": "hgb",
        "grid": {"max_leaf_nodes": [15, 31, 63], "learning_rate": [0.05, 0.1],
                 "min_samples_leaf": [20, 100]},
    },
}

DIAG_FIELDS = ["strategy", "n_splits", "n_candidates", "cv_ap_mean", "cv_ap_std",
               "test_pr_auc", "test_roc_auc", "best_params", "fit_time_s"]
TUNE_FIELDS = ["model", "select_cv", "val_ap", "purged_cv_ap", "purged_cv_std",
               "best_params", "final_n_rounds", "test_pr_auc", "test_roc_auc",
               "recall_global", "base_pr_auc", "fit_time_s", "peak_rss_mb",
               "status", "error"]


# --------------------------------------------------------------------------- #
# shared loaders
# --------------------------------------------------------------------------- #
def _load():
    a = C.load_frozen(mmap=True)
    times_tr, _ = C.load_win_time()
    return (a["X_train"], a["X_test"], np.asarray(a["y_train"]), np.asarray(a["y_test"]),
            np.asarray(a["states_test"]), np.asarray(a["job_ids_train"]), times_tr)


def _temporal_holdout(times_tr: np.ndarray, job_ids_tr: np.ndarray, frac: float = 0.15):
    """Last `frac` of TRAIN jobs by end-time -> early-stopping eval set. Returns
    boolean masks (fit_mask, hold_mask) over train rows; jobs are never split.

    The holdout is TEMPORAL (latest jobs) on purpose: we proved a same-period
    (grouped-random) monitor overfits the future -- more trees keep improving its
    AP while degrading the temporal-test AP. Only a temporal monitor selects the
    number of rounds that generalizes forward.
    """
    b = pd.DataFrame({"job": job_ids_tr, "t": times_tr}).groupby("job")["t"].max()
    b = b.sort_values()
    n_hold = int(len(b) * frac)
    hold_jobs = set(b.index[-n_hold:].tolist())
    hold_mask = np.fromiter((j in hold_jobs for j in job_ids_tr), dtype=bool, count=len(job_ids_tr))
    return ~hold_mask, hold_mask


def _grid_candidates(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _early_stop_val(kind, params, spw, Xf, yf, Xh, yh):
    """Fit one candidate with round-wise early stopping on the temporal holdout.
    Returns (best_n_rounds, val_ap) where val_ap = AP on the holdout at that round,
    computed uniformly with sklearn so all model families are comparable."""
    if kind == "lgbm":
        from lightgbm import early_stopping, log_evaluation
        est = _lgbm(n_estimators=MAX_ROUNDS, scale_pos_weight=spw,
                    metric="average_precision", **params)
        est.fit(Xf, yf, eval_set=[(Xh, yh)],
                callbacks=[early_stopping(ES_PATIENCE, first_metric_only=True, verbose=False),
                           log_evaluation(0)])
        n = int(est.best_iteration_ or MAX_ROUNDS)
        val = float(average_precision_score(yh, est.predict_proba(Xh, num_iteration=n)[:, 1]))
        return n, val
    if kind == "catboost":
        est = _catb(iterations=MAX_ROUNDS, eval_metric="PRAUC", **params)
        est.fit(Xf, yf, eval_set=(Xh, yh), early_stopping_rounds=ES_PATIENCE,
                use_best_model=True, verbose=0)
        n = int(est.get_best_iteration() or est.tree_count_)
        val = float(average_precision_score(yh, est.predict_proba(Xh)[:, 1]))
        return n, val
    # hgb: warm-start staged over max_iter; pick the stage with best holdout AP
    from sklearn.ensemble import HistGradientBoostingClassifier
    est = HistGradientBoostingClassifier(class_weight="balanced", random_state=C.SEED,
                                         warm_start=True, early_stopping=False, **params)
    best_n, best_val = HGB_STAGES[0], -1.0
    for mi in HGB_STAGES:
        est.set_params(max_iter=mi)
        est.fit(Xf, yf)
        v = float(average_precision_score(yh, est.predict_proba(Xh)[:, 1]))
        if v > best_val:
            best_val, best_n = v, mi
    return best_n, best_val


def _final_estimator(kind, params, n_rounds, spw):
    """Build the final estimator at the chosen structure + round count (for a
    full-train refit)."""
    if kind == "lgbm":
        return _lgbm(n_estimators=n_rounds, scale_pos_weight=spw, **params)
    if kind == "catboost":
        return _catb(iterations=n_rounds, **params)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=C.SEED,
                                          max_iter=n_rounds, early_stopping=False, **params)


def _base_pr_auc(model: str) -> float:
    df = pd.read_csv(C.BENCH_RESULTS_CSV)
    r = df[(df["model"] == model) & (df["pass"] == 1)].drop_duplicates("model", keep="last")
    return float(r["pr_auc"].iloc[0]) if len(r) else float("nan")


def _purged_cv_ap(kind, params, n_rounds, spw, X_tr, y_tr, splits) -> tuple[float, float]:
    """Validation-only: AP of the chosen config across the purged folds (robustness)."""
    aps = []
    for tr, te in splits:
        est = _final_estimator(kind, params, n_rounds, spw)
        est.fit(np.ascontiguousarray(X_tr[tr]), y_tr[tr])
        aps.append(average_precision_score(y_tr[te], est.predict_proba(np.ascontiguousarray(X_tr[te]))[:, 1]))
        del est
        gc.collect()
    return float(np.mean(aps)), float(np.std(aps))


# --------------------------------------------------------------------------- #
# Phase 4 - diagnosis
# --------------------------------------------------------------------------- #
def diagnostic() -> None:
    C.ensure_dirs()
    if C.CV_COMPARISON_CSV.exists():
        C.CV_COMPARISON_CSV.unlink()
    X_tr, X_te, y_tr, y_te, states_te, groups_tr, times_tr = _load()
    spw = _spw()
    print("=" * 78)
    print("PHASE 4 - CV DIAGNOSIS (LightGBM small grid under each strategy)")
    print("=" * 78)
    print(f"{'strategy':<17}{'folds':>6}{'CV-AP':>10}{'test-PR-AUC':>14}{'inflation':>11}")
    print("-" * 78)

    for name in CV.STRATEGIES:
        splits = CV.make_splits(name, y=y_tr, groups=groups_tr, times=times_tr)
        est = _lgbm(n_estimators=DIAG_N_ESTIMATORS, scale_pos_weight=spw)
        gs = GridSearchCV(est, DIAG_GRID, scoring="average_precision",
                          cv=splits, n_jobs=1, refit=True)
        t0 = time.perf_counter()
        gs.fit(X_tr, y_tr)
        dt = round(time.perf_counter() - t0, 1)

        proba = gs.best_estimator_.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, proba, states_te)
        cv_ap = float(gs.best_score_)
        cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
        infl = cv_ap / m["pr_auc"] if m["pr_auc"] else float("nan")

        append_row(C.CV_COMPARISON_CSV, {
            "strategy": name, "n_splits": len(splits),
            "n_candidates": len(gs.cv_results_["params"]),
            "cv_ap_mean": cv_ap, "cv_ap_std": cv_std,
            "test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"],
            "best_params": json.dumps(gs.best_params_), "fit_time_s": dt,
        }, DIAG_FIELDS)
        print(f"{name:<17}{len(splits):>6}{cv_ap:>10.4f}{m['pr_auc']:>14.4f}{infl:>10.1f}x")
        del gs, proba
        gc.collect()

    print("-" * 78)
    print(f"  baseline (test prevalence) = 0.0403   |   written: {C.CV_COMPARISON_CSV.name}")
    print("  Read: honest CV-AP should sit near test-PR-AUC; 'stratified' should not.")


# --------------------------------------------------------------------------- #
# Phase 5 - honest re-tune
# --------------------------------------------------------------------------- #
def retune(models: list[str], cv_name: str = "purged") -> None:
    C.ensure_dirs()
    X_tr, X_te, y_tr, y_te, states_te, groups_tr, times_tr = _load()
    spw = _spw()
    fit_mask, hold_mask = _temporal_holdout(times_tr, groups_tr)
    Xf = np.ascontiguousarray(X_tr[fit_mask]); yf = y_tr[fit_mask]
    Xh = np.ascontiguousarray(X_tr[hold_mask]); yh = y_tr[hold_mask]
    purged = CV.make_splits(cv_name, y=y_tr, groups=groups_tr, times=times_tr)
    print("=" * 82)
    print("PHASE 5 - HONEST RE-TUNE")
    print(f"  select rounds+structure by early stopping on a TEMPORAL holdout "
          f"({int(hold_mask.sum()):,} rows, latest {hold_mask.mean()*100:.0f}% of train jobs)")
    print(f"  robustness check: '{cv_name}' CV ({len(purged)} folds) on the chosen config")
    print("=" * 82)

    for name in models:
        cfg = TUNE[name]; kind = cfg["kind"]
        candidates = _grid_candidates(cfg["grid"])
        try:
            with PeakMemorySampler() as mem:
                t0 = time.perf_counter()
                # 1) select structure + rounds on the temporal holdout ---------- #
                best = None
                for params in candidates:
                    n, val = _early_stop_val(kind, params, spw, Xf, yf, Xh, yh)
                    if best is None or val > best[2]:
                        best = (params, n, val)
                    print(f"      [{name}] {params} -> rounds={n} val_AP={val:.4f}")
                best_params, n_rounds, val_ap = best

                # 2) robustness: chosen config under the purged CV -------------- #
                pcv_ap, pcv_std = _purged_cv_ap(kind, best_params, n_rounds, spw,
                                                X_tr, y_tr, purged)

                # 3) refit on FULL train at chosen structure+rounds; eval on test #
                est = _final_estimator(kind, best_params, n_rounds, spw)
                est.fit(X_tr, y_tr)
                proba = est.predict_proba(X_te)[:, 1]
                dt = round(time.perf_counter() - t0, 1)
            m = evaluate(y_te, proba, states_te)
            base = _base_pr_auc(name)

            append_row(C.TUNING_RESULTS_CSV, {
                "model": name, "select_cv": "temporal_holdout", "val_ap": val_ap,
                "purged_cv_ap": pcv_ap, "purged_cv_std": pcv_std,
                "best_params": json.dumps(best_params), "final_n_rounds": n_rounds,
                "test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"],
                "recall_global": m["recall_global"], "base_pr_auc": base,
                "fit_time_s": dt, "peak_rss_mb": round(mem.peak_mb, 1),
                "status": "ok", "error": "",
            }, TUNE_FIELDS)
            import joblib
            joblib.dump(est, C.GRID_DIR / f"{name}_tuned.joblib")
            np.save(C.GRID_DIR / f"{name}_tuned_test_proba.npy", proba.astype(np.float32))

            delta = m["pr_auc"] - base
            print(f"  {name:<32} TEST PR-AUC={m['pr_auc']:.4f} (base {base:.4f}, "
                  f"{'+' if delta >= 0 else ''}{delta:.4f})  ROC={m['roc_auc']:.4f}  "
                  f"rounds={n_rounds}  purgedCV={pcv_ap:.4f}  ({dt:.0f}s, peak {mem.peak_mb:.0f}MB)")
            print(f"      chosen: {best_params}")
            del est, proba
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            append_row(C.TUNING_RESULTS_CSV, {
                "model": name, "select_cv": "temporal_holdout", "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"}, TUNE_FIELDS)

    del Xf, Xh
    gc.collect()
    print("=" * 82)
    print(f"  written: {C.TUNING_RESULTS_CSV}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic", action="store_true", help="Phase 4 CV diagnosis")
    ap.add_argument("--retune", action="store_true", help="Phase 5 honest re-tune")
    ap.add_argument("--cv", default="purged", choices=CV.STRATEGIES, help="honest CV for re-tune")
    ap.add_argument("--models", default="LightGBM,CatBoost,HistGradientBoostingClassifier")
    args = ap.parse_args()
    if args.diagnostic:
        diagnostic()
    if args.retune:
        retune([m.strip() for m in args.models.split(",") if m.strip()], args.cv)
    if not (args.diagnostic or args.retune):
        ap.print_help()


if __name__ == "__main__":
    main()
