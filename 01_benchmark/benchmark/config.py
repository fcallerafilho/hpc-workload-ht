"""Central configuration: paths, seed, feature/state definitions, constants.

Single source of truth. Every other module imports from here so that the 8
features, the window size, the split fraction, and the state encoding are
defined exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths. This package lives in 01_benchmark/benchmark/, so its parent is the
# session folder (artifacts and results live there) and its grandparent is the
# project root (where dataset/ lives).
# --------------------------------------------------------------------------- #
SESSION_DIR = Path(__file__).resolve().parent.parent
ROOT = SESSION_DIR.parent
DATASET_PATH = ROOT / "dataset" / "prom_slurm_joined" / "prom_slurm_joined.parquet"
ARTIFACTS_DIR = SESSION_DIR / "artifacts"
RESULTS_DIR = SESSION_DIR / "results"
PASS2_DIR = RESULTS_DIR / "pass2_grid_results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
LOG_FILE = RESULTS_DIR / "benchmark.log"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Pipeline constants (authoritative source: TCC_Dados_ML.pdf)
# --------------------------------------------------------------------------- #
WINDOW_SIZE = 40          # 40 snapshots x 15 s = 10 min window
PRE_FAILURE_HOURS = 2     # label=1 within the 2 h before a failure job's end_date
TRAIN_FRACTION = 0.8      # temporal 80/20 split by job end_date
STRIDE = 1                # sliding window advances 1 snapshot (39/40 overlap)

# The 8 selected metrics (Skrzeczek 2024). Column order here defines feature order.
FEATURES = [
    "node_memory_Active_bytes",
    "node_disk_written_bytes_total_sum",
    "node_netstat_Tcp_InErrs",
    "node_forks_total",
    "nvidia_gpu_temperature_celsius_mean",
    "nvidia_gpu_power_usage_milliwatts_mean",
    "nvidia_gpu_memory_used_bytes_sum",
    "nvidia_gpu_fanspeed_percent_mean",
]

# Scaffold columns needed for labeling / splitting / grouping (not features).
SCAFFOLD = ["slurm_id", "state", "end_date", "timestamp"]

# Aggregations applied over each 40-snapshot window, in concatenation order.
# The windowed matrix is [min(8) | max(8) | mean(8) | std(8)] -> 32 columns.
AGG_ORDER = ["min", "max", "mean", "std"]

# --------------------------------------------------------------------------- #
# Job states
# --------------------------------------------------------------------------- #
# Raw Slurm state strings as they appear in the parquet.
STATE_COMPLETED = "COMPLETED"
FAILURE_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

# Stable integer encoding for the states_*.npy artifacts (int8 codes + this map).
# Kept small and fixed so codes are reproducible across runs.
STATE_CODES = {
    "COMPLETED": 0,
    "TIMEOUT": 1,
    "FAILED": 2,
    "OUT_OF_MEMORY": 3,
    "NODE_FAIL": 4,
}
CODE_TO_STATE = {v: k for k, v in STATE_CODES.items()}

# Reporting: the raw state OUT_OF_MEMORY is displayed as OOM in output columns.
# (Thesis vocabulary: "OOM" == the OUT_OF_MEMORY job state, NOT machine RAM.)
STATE_DISPLAY = {
    "TIMEOUT": "TIMEOUT",
    "FAILED": "FAILED",
    "OUT_OF_MEMORY": "OOM",
    "NODE_FAIL": "NODE_FAIL",
    "COMPLETED": "COMPLETED",
}
# Failure states reported for per-state recall (display order).
REPORT_FAILURE_STATES = ["TIMEOUT", "FAILED", "OUT_OF_MEMORY", "NODE_FAIL"]
# All states reported for per-state mean_proba (adds COMPLETED as a diagnostic).
REPORT_ALL_STATES = REPORT_FAILURE_STATES + ["COMPLETED"]

# --------------------------------------------------------------------------- #
# Derived helpers
# --------------------------------------------------------------------------- #
def feature_names() -> list[str]:
    """The 32 windowed feature names in column order (matches AGG_ORDER blocks)."""
    return [f"{feat}_{agg}" for agg in AGG_ORDER for feat in FEATURES]


# Artifact file names (under ARTIFACTS_DIR).
ARTIFACTS = {
    "X_train": "X_train.npy",
    "X_test": "X_test.npy",
    "y_train": "y_train.npy",
    "y_test": "y_test.npy",
    "states_train": "states_train.npy",
    "states_test": "states_test.npy",
    "job_ids_train": "job_ids_train.npy",
    "job_ids_test": "job_ids_test.npy",
    "feature_names": "feature_names.json",
    "state_names": "state_names.json",
    "scaler": "scaler.joblib",
}


def artifact_path(key: str) -> Path:
    return ARTIFACTS_DIR / ARTIFACTS[key]


def artifacts_exist() -> bool:
    """True if the full set of pipeline outputs is already on disk."""
    return all(artifact_path(k).exists() for k in ARTIFACTS)


def load_state_names() -> dict[int, str]:
    """Load {code: raw_state} from the artifact (falls back to config default)."""
    p = artifact_path("state_names")
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            return {int(k): v for k, v in json.load(fh).items()}
    return dict(CODE_TO_STATE)
