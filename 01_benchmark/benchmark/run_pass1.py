"""Pass 1: train every base model once, rank by PR-AUC.

    python -m benchmark.run_pass1

Each model is isolated in try/except so one failure (missing dep, RAM
exhaustion on the 16 GB laptop, non-convergence) never aborts the benchmark.
Results are appended to results/results.csv immediately after each model.
"""
from __future__ import annotations

import gc
import json
import time
import traceback

import joblib
import numpy as np

from . import config
from .evaluation import METRIC_COLUMNS, evaluate
from .models import ModelSpec, get_base_models
from .utils import PeakMemorySampler, append_row, get_logger, set_seed

log = get_logger()

ID_COLUMNS = ["model", "family", "pass", "params", "train_time_s", "predict_time_s"]
DIAG_COLUMNS = ["peak_rss_mb", "status", "error"]
FIELDNAMES = ID_COLUMNS + METRIC_COLUMNS + DIAG_COLUMNS


def _load_artifacts():
    if not config.artifacts_exist():
        raise SystemExit(
            "Artifacts missing. Run:  python -m benchmark.data_pipeline")
    log.info("Loading artifacts (X as read-only memmap)...")
    X_train = np.load(config.artifact_path("X_train"), mmap_mode="r")
    X_test = np.load(config.artifact_path("X_test"), mmap_mode="r")
    y_train = np.load(config.artifact_path("y_train"))
    y_test = np.load(config.artifact_path("y_test"))
    states_test = np.load(config.artifact_path("states_test"))
    scaler = joblib.load(config.artifact_path("scaler"))
    return X_train, X_test, y_train, y_test, states_test, scaler


def _run_one(spec: ModelSpec, X_train, X_test, y_train, y_test, states_test, scaler) -> dict:
    row = {"model": spec.name, "family": spec.family, "pass": 1,
           "params": json.dumps(spec.params or {}), "status": "ok", "error": ""}

    # Scaled models consume StandardScaler-transformed float32 copies.
    if spec.needs_scaling:
        Xtr = np.ascontiguousarray(scaler.transform(X_train), dtype=np.float32)
        Xte = np.ascontiguousarray(scaler.transform(X_test), dtype=np.float32)
    else:
        Xtr, Xte = X_train, X_test

    with PeakMemorySampler() as mem:
        t0 = time.perf_counter()
        spec.estimator.fit(Xtr, y_train)
        row["train_time_s"] = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        proba = spec.estimator.predict_proba(Xte)[:, 1]
        row["predict_time_s"] = round(time.perf_counter() - t1, 2)

    row["peak_rss_mb"] = round(mem.peak_mb, 1)
    row.update(evaluate(y_test, proba, states_test))

    log.info("%-32s  PR-AUC=%.4f  ROC-AUC=%.4f  recall=%.3f  train=%.1fs  peak=%.0f MB",
             spec.name, row["pr_auc"], row["roc_auc"], row["recall_global"],
             row["train_time_s"], row["peak_rss_mb"])

    # free scaled copies / prediction before the next model
    if spec.needs_scaling:
        del Xtr, Xte
    del proba
    return row


def run(only: list[str] | None = None) -> None:
    set_seed()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, states_test, scaler = _load_artifacts()

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    log.info("Train windows: %s  (pos=%s, neg=%s)  scale_pos_weight=%.4f",
             f"{len(y_train):,}", f"{n_pos:,}", f"{n_neg:,}", scale_pos_weight)

    specs = get_base_models(scale_pos_weight)
    if only:
        wanted = {m.strip() for m in only}
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            log.warning("Unknown model name(s) ignored: %s", ", ".join(sorted(missing)))
    log.info("Pass 1: %d model(s) -> %s", len(specs), config.RESULTS_CSV)

    for i, spec in enumerate(specs, 1):
        log.info("[%2d/%2d] %s", i, len(specs), spec.name)
        try:
            row = _run_one(spec, X_train, X_test, y_train, y_test, states_test, scaler)
        except Exception as exc:  # noqa: BLE001 - isolate each model
            tb = traceback.format_exc()
            log.error("%s FAILED: %s", spec.name, exc)
            log.error(tb)
            row = {"model": spec.name, "family": spec.family, "pass": 1,
                   "params": json.dumps(spec.params or {}),
                   "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        append_row(config.RESULTS_CSV, row, FIELDNAMES)
        # release the estimator (and any large internal state) before the next model
        spec.estimator = None
        gc.collect()

    _print_ranking()


def _print_ranking() -> None:
    import pandas as pd

    df = pd.read_csv(config.RESULTS_CSV)
    df = df[df["pass"] == 1].drop_duplicates("model", keep="last")
    ok = df[df["status"] == "ok"].sort_values("pr_auc", ascending=False)

    log.info("=" * 78)
    log.info("PASS 1 RANKING by PR-AUC")
    log.info("%-32s %8s %8s %8s %9s", "model", "PR-AUC", "ROC-AUC", "recall", "train_s")
    log.info("-" * 78)
    for rank, (_, r) in enumerate(ok.iterrows(), 1):
        marker = "  <-- top" if rank <= 4 else ""
        log.info("%-32s %8.4f %8.4f %8.3f %9.1f%s", r["model"], r["pr_auc"],
                 r["roc_auc"], r["recall_global"], r["train_time_s"], marker)
    failed = df[df["status"] != "ok"]
    if len(failed):
        log.info("-" * 78)
        for _, r in failed.iterrows():
            log.info("FAILED: %-30s %s", r["model"], r.get("error", ""))
    log.info("=" * 78)
    if len(ok) >= 3:
        log.info("Top-4 candidates for Pass 2: %s", ", ".join(ok["model"].head(4)))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Pass 1: train base models, rank by PR-AUC.")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of model names to (re)run")
    args = ap.parse_args()
    run(only=[m for m in args.only.split(",")] if args.only else None)


if __name__ == "__main__":
    main()
