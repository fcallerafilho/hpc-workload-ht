"""Configuration for the session-2 `cv_strategies` work.

Thin by design: the constants that *describe the frozen artifacts* (feature
order, state codes, the 8 metrics, the seed, the window size, and the paths to
the session-1 `.npy` files) are **re-exported read-only** from `benchmark.config`
rather than re-declared here. That guarantees this session can never drift from
the arrays saved in session 1.

Everything genuinely new to this session (the per-window time-key artifact, the
results/plots/docs locations) is defined below.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Make the session-1 `benchmark` package importable no matter how we are run
# (`python -m cv_strategies.xxx` from 02_cv_strategies/, a notebook, or a script).
# --------------------------------------------------------------------------- #
CV_DIR = Path(__file__).resolve().parent.parent        # session-2 folder
ROOT = CV_DIR.parent                                   # project root
BENCH_DIR = ROOT / "01_benchmark"                      # session-1 folder: holds the
if str(BENCH_DIR) not in sys.path:                     # importable `benchmark` package
    sys.path.insert(0, str(BENCH_DIR))

# Read-only reuse of the session-1 constants (NOT edited, only imported). ----- #
from benchmark import config as bench          # noqa: E402

SEED = bench.SEED
WINDOW_SIZE = bench.WINDOW_SIZE
PRE_FAILURE_HOURS = bench.PRE_FAILURE_HOURS
TRAIN_FRACTION = bench.TRAIN_FRACTION
FEATURES = bench.FEATURES
SCAFFOLD = bench.SCAFFOLD
AGG_ORDER = bench.AGG_ORDER
FAILURE_STATES = bench.FAILURE_STATES
STATE_CODES = bench.STATE_CODES
CODE_TO_STATE = bench.CODE_TO_STATE
STATE_DISPLAY = bench.STATE_DISPLAY
REPORT_FAILURE_STATES = bench.REPORT_FAILURE_STATES
REPORT_ALL_STATES = bench.REPORT_ALL_STATES
feature_names = bench.feature_names

# Frozen session-1 locations (reused, never written to). --------------------- #
DATASET_PATH = bench.DATASET_PATH               # the 10 GB parquet
FROZEN_ARTIFACTS_DIR = bench.ARTIFACTS_DIR      # ../artifacts (X_*, y_*, states_*, job_ids_*)
frozen_artifact_path = bench.artifact_path      # callable: key -> Path in ../artifacts
BENCH_RESULTS_CSV = bench.RESULTS_CSV           # session-1 results (Dummy row etc.)

# --------------------------------------------------------------------------- #
# New session-2 locations (this folder)
# --------------------------------------------------------------------------- #
ARTIFACTS_DIR = CV_DIR / "artifacts"            # win_time_{train,test}.npy live here
RESULTS_DIR = CV_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
GRID_DIR = RESULTS_DIR / "pass2_grid_results"
NOTEBOOKS_DIR = CV_DIR / "notebooks"
DOCS_DIR = CV_DIR / "docs"
LOG_FILE = RESULTS_DIR / "cv_strategies.log"

# New artifacts / outputs
WIN_TIME = {
    "win_time_train": ARTIFACTS_DIR / "win_time_train.npy",
    "win_time_test": ARTIFACTS_DIR / "win_time_test.npy",
}
IMBALANCE_JSON = RESULTS_DIR / "imbalance.json"
CV_COMPARISON_CSV = RESULTS_DIR / "cv_comparison.csv"
TUNING_RESULTS_CSV = RESULTS_DIR / "tuning_results.csv"
OPERATING_POINTS_CSV = RESULTS_DIR / "operating_points.csv"
ENSEMBLE_CALIB_CSV = RESULTS_DIR / "ensemble_calibration.csv"
REPORT_HTML = RESULTS_DIR / "report.html"


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, RESULTS_DIR, PLOTS_DIR, GRID_DIR, NOTEBOOKS_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Convenience loaders for the frozen artifacts (X as read-only memmap)
# --------------------------------------------------------------------------- #
def load_frozen(mmap: bool = True):
    """Load the reusable session-1 arrays. X_* are memmapped read-only so they
    are shared across processes and never duplicated (16 GB budget)."""
    import numpy as np
    mm = "r" if mmap else None
    return {
        "X_train": np.load(frozen_artifact_path("X_train"), mmap_mode=mm),
        "X_test": np.load(frozen_artifact_path("X_test"), mmap_mode=mm),
        "y_train": np.load(frozen_artifact_path("y_train")),
        "y_test": np.load(frozen_artifact_path("y_test")),
        "states_train": np.load(frozen_artifact_path("states_train")),
        "states_test": np.load(frozen_artifact_path("states_test")),
        "job_ids_train": np.load(frozen_artifact_path("job_ids_train")),
        "job_ids_test": np.load(frozen_artifact_path("job_ids_test")),
    }


def load_win_time():
    """Load the per-window end-time arrays built by timekey.py (int64 ns)."""
    import numpy as np
    return (np.load(WIN_TIME["win_time_train"]), np.load(WIN_TIME["win_time_test"]))


def win_time_exists() -> bool:
    return all(p.exists() for p in WIN_TIME.values())
