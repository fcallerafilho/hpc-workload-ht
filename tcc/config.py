"""Paths and fixed constants of the study.

Every value here was fixed before any model was fitted. The dated reasons are in DECISOES.md
(entries P1 to P13); the entry that fixes each value is given in the comment.
"""
import os
from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("TCC_DATASET", ROOT / "dataset"))
M100 = DATASET / "m100"                          # M100 ExaData, extracted: year_month=20-05 ... 20-10
PM100_JOBS = DATASET / "pm100" / "job_table.parquet"
DATA = ROOT / "data"                             # derived data written by the notebooks (not versioned)
RESULTS = ROOT / "results"                       # the paper's tables as CSV and the claim checks (versioned)
for _d in (DATA, RESULTS / "claims"):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ fixed constants
SEED = 42
START, END = "2020-05-06", "2020-10-12 23:00"    # cohort limits, UTC (P1)
MONTHS = ["20-05", "20-06", "20-07", "20-08", "20-09", "20-10"]
WINDOW_S = 1200                                  # the analysis window: first 20 minutes of the job
MIN_RUNTIME_S = 1200                             # a job must run at least as long as the window
FLOOR = 100                                      # minimum positives per test set (P2)
N_BOOT = 1000                                    # user-level bootstrap resamples (P10)
N_PLACEBOS = 20                                  # placebo groups (P6)

STATES = ["COMPLETED", "FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "TIMEOUT"]
STATES_A = ["FAILED", "OUT_OF_MEMORY", "NODE_FAIL"]   # target (a); target (b) is TIMEOUT

# ------------------------------------------------------------------ telemetry
# ExaData metric name -> plugin.  ganglia_pub = ganglia server (gmond); ipmi_pub = node BMC.
CHANNELS = {
    "mem_total": "ganglia_pub", "mem_free": "ganglia_pub", "mem_cached": "ganglia_pub",
    "mem_buffers": "ganglia_pub", "swap_free": "ganglia_pub", "cpu_wio": "ganglia_pub",
    "proc_run": "ganglia_pub", "load_one": "ganglia_pub",
    "gpu0_core_temp": "ipmi_pub", "gpu1_core_temp": "ipmi_pub", "gpu3_core_temp": "ipmi_pub",
    "gpu4_core_temp": "ipmi_pub", "gpu0_mem_temp": "ipmi_pub", "gpu4_mem_temp": "ipmi_pub",
    "fan0_0": "ipmi_pub", "total_power": "ipmi_pub", "ambient": "ipmi_pub",
    "gv100card0": "ipmi_pub", "gv100card3": "ipmi_pub",
}


def source(channel):
    return "ganglia" if CHANNELS[channel] == "ganglia_pub" else "IPMI"


# The ten metric groups (paper, Table "Metric groups").  The ORDER matters: placebo k copies
# group number k mod 10 (P6), so changing it changes the placebos.
GROUPS = {
    "mem_active":    ["mem_total", "mem_free", "mem_cached", "mem_buffers"],
    "swap_free":     ["swap_free"],
    "cpu_wio":       ["cpu_wio"],
    "proc_run":      ["proc_run"],
    "load_one":      ["load_one"],
    "gpu_core_temp": ["gpu0_core_temp", "gpu1_core_temp", "gpu3_core_temp", "gpu4_core_temp"],
    "gpu_mem_temp":  ["gpu0_mem_temp", "gpu4_mem_temp"],
    "fan":           ["fan0_0"],
    "node_power":    ["total_power"],
    "inlet_temp":    ["ambient"],
}
IPMI_GROUPS = [g for g, chans in GROUPS.items() if source(chans[0]) == "IPMI"]

# The two regimes of Section II-A (used by the pre-defined readings, P11).
HARDWARE_REGULATED = ["gpu_core_temp", "gpu_mem_temp", "fan", "inlet_temp"]
LOAD_DRIVEN = ["mem_active", "swap_free", "cpu_wio", "proc_run", "load_one", "node_power"]

STATS = ["min", "max", "mean", "std"]


def group_cols(group):
    """The four feature columns of a group (or of a placebo)."""
    return [f"{group}__{s}" for s in STATS]


# B0: what the user requests at submission (P4).  qos and partition are one-hot encoded.
B0_NUMERIC = ["time_limit", "mem_req", "num_nodes_req", "num_cores_req", "num_gpus_req",
              "num_tasks", "cores_per_task"]
