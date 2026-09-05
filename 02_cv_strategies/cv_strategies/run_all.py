"""Reproduce the whole Session-2 pipeline (phases 1-8), skipping finished steps.

    python -m cv_strategies.run_all            # skip steps whose outputs exist
    python -m cv_strategies.run_all --force     # rebuild everything
    python -m cv_strategies.run_all --cv purged # honest CV for the re-tune

Heavy steps (diagnosis ~45 min, re-tune ~60-90 min) dominate the runtime.
"""
from __future__ import annotations

import argparse

from . import config as C


def run(force: bool = False, cv: str = "purged") -> None:
    C.ensure_dirs()

    # 1. exact baseline
    if force or not C.IMBALANCE_JSON.exists():
        from . import imbalance
        imbalance.main()

    # 2. window-time artifact
    if force or not C.win_time_exists():
        from . import timekey
        timekey.build()

    # 3. CV self-tests (fast, always)
    from . import cv as CVmod
    CVmod._selftest()

    # 4. CV diagnosis
    if force or not C.CV_COMPARISON_CSV.exists():
        from . import tuning
        tuning.diagnostic()

    # 5. honest re-tune
    if force or not C.TUNING_RESULTS_CSV.exists():
        from . import tuning
        tuning.retune(["LightGBM", "CatBoost", "HistGradientBoostingClassifier"], cv)

    # 6. operating points
    if force or not C.OPERATING_POINTS_CSV.exists():
        from . import evaluation_ext
        evaluation_ext.operating_points()

    # 7. ensemble + calibration
    if force or not C.ENSEMBLE_CALIB_CSV.exists():
        from . import ensemble_calib
        ensemble_calib.main()

    # 8. report
    from . import report
    report.build()
    print("\nAll phases complete. Open:", C.REPORT_HTML)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Session-2 pipeline end to end.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cv", default="purged")
    args = ap.parse_args()
    run(force=args.force, cv=args.cv)


if __name__ == "__main__":
    main()
