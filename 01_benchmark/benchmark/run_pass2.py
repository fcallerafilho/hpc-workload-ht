"""Pass 2: GridSearchCV for the top models from Pass 1.

    python -m benchmark.run_pass2                     # top 4 by PR-AUC (with grids)
    python -m benchmark.run_pass2 --top 3
    python -m benchmark.run_pass2 --models LightGBM,XGBoost

GridSearchCV runs with n_jobs=1 by design: each estimator already uses all cores
(threaded RF / multithreaded boosters), so candidates train one-at-a-time and the
1.09 GB training matrix is never duplicated across worker processes. This trades
speed for staying inside the 16 GB laptop RAM budget -- Pass 2 is the stage most
likely to exhaust RAM.  StratifiedKFold(3, shuffle, seed) per the spec; note that
overlapping same-job windows make the CV score optimistic vs. the held-out test
(reported metrics are always on the temporal test set).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config
from .evaluation import METRIC_COLUMNS, evaluate
from .grids import get_grids
from .run_pass1 import FIELDNAMES
from .utils import PeakMemorySampler, append_row, get_logger, set_seed

log = get_logger()


def _select_models(top: int, explicit: str | None, grids: dict) -> list[str]:
    if explicit:
        wanted = [m.strip() for m in explicit.split(",") if m.strip()]
    else:
        df = pd.read_csv(config.RESULTS_CSV)
        df = df[(df["pass"] == 1) & (df["status"] == "ok")].drop_duplicates(
            "model", keep="last").sort_values("pr_auc", ascending=False)
        wanted = list(df["model"].head(top))
        log.info("Top-%d from Pass 1 by PR-AUC: %s", top, ", ".join(wanted))

    selected, skipped = [], []
    for m in wanted:
        (selected if m in grids else skipped).append(m)
    if skipped:
        log.info("No grid defined (skipped): %s", ", ".join(skipped))
    return selected


def _n_candidates(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def run(top: int = 4, explicit: str | None = None) -> None:
    set_seed()
    config.PASS2_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(config.artifact_path("X_train"), mmap_mode="r")
    X_test = np.load(config.artifact_path("X_test"), mmap_mode="r")
    y_train = np.load(config.artifact_path("y_train"))
    y_test = np.load(config.artifact_path("y_test"))
    states_test = np.load(config.artifact_path("states_test"))

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    grids = get_grids(float(scale_pos_weight))
    selected = _select_models(top, explicit, grids)
    if not selected:
        log.info("No grid-able models selected for Pass 2.")
        return

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)

    for name in selected:
        spec = grids[name]
        estimator = spec.make_estimator()
        param_grid = spec.param_grid

        # Scaled models: scale inside each CV fold via a pipeline.
        if spec.needs_scaling:
            from sklearn.preprocessing import StandardScaler
            estimator = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        n_fits = _n_candidates(spec.param_grid) * cv.get_n_splits()
        log.info("[Pass 2] %s: %d candidates x %d folds = %d fits",
                 name, _n_candidates(spec.param_grid), cv.get_n_splits(), n_fits)

        gs = GridSearchCV(estimator, param_grid, scoring="average_precision",
                          cv=cv, n_jobs=1, refit=True, verbose=2)
        try:
            with PeakMemorySampler() as mem:
                t0 = time.perf_counter()
                gs.fit(X_train, y_train)
                train_time = round(time.perf_counter() - t0, 2)

                t1 = time.perf_counter()
                proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
                predict_time = round(time.perf_counter() - t1, 2)
        except Exception as exc:  # noqa: BLE001
            log.error("%s grid FAILED: %s", name, exc)
            log.error(traceback.format_exc())
            append_row(config.RESULTS_CSV,
                       {"model": name, "family": "pass2", "pass": 2,
                        "params": json.dumps(param_grid), "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"}, FIELDNAMES)
            continue

        row = {"model": name, "family": "pass2", "pass": 2,
               "params": json.dumps(gs.best_params_, default=str),
               "train_time_s": train_time, "predict_time_s": predict_time,
               "peak_rss_mb": round(mem.peak_mb, 1), "status": "ok", "error": ""}
        row.update(evaluate(y_test, proba, states_test))
        append_row(config.RESULTS_CSV, row, FIELDNAMES)

        # persist grid details
        pd.DataFrame(gs.cv_results_).to_csv(
            config.PASS2_DIR / f"{name}_cv_results.csv", index=False)
        with open(config.PASS2_DIR / f"{name}_best_params.json", "w", encoding="utf-8") as fh:
            json.dump(gs.best_params_, fh, indent=2, default=str)

        log.info("%-32s  best CV AP=%.4f  |  TEST PR-AUC=%.4f  ROC-AUC=%.4f  (%.0f s, peak %.0f MB)",
                 name, gs.best_score_, row["pr_auc"], row["roc_auc"], train_time, row["peak_rss_mb"])
        log.info("  best params: %s", gs.best_params_)

        del gs, proba
        gc.collect()

    log.info("Pass 2 complete. Details in %s", config.PASS2_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass 2 grid search on top-K models.")
    ap.add_argument("--top", type=int, default=4, help="number of top Pass-1 models")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model names (overrides --top)")
    args = ap.parse_args()
    run(top=args.top, explicit=args.models)


if __name__ == "__main__":
    main()
