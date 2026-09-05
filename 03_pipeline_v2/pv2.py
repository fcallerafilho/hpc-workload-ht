"""pv2 - the one shared module of the Session-3 pipeline (``pipeline_v2``).

Everything in this session is built **from scratch**: nothing is imported from
``benchmark/`` (Session 1) or ``cv_strategies/`` (Session 2), and no ``.npy``
produced by those sessions is read. Their *numbers* are quoted in the write-up
for comparison, but not their code and not their arrays.

Why a module at all, when the work lives in notebooks? Because six notebooks
must agree, to the bit, on the paths, the metric list, the window mechanics and
the evaluation protocol. Duplicating those in every notebook is how a pipeline
silently drifts. So: **one** module holding the machinery, and notebooks that
tell the story and call into it.

--------------------------------------------------------------------------- #
The time notions carried over from Session 2 (non-negotiable invariants)
--------------------------------------------------------------------------- #
1. Prediction is **per window**, never per job. A window is a short slice of one
   job's telemetry; the dataset is never collapsed to one row per job.
2. Labels are **time-based**: a snapshot is positive iff its job ends in a
   failure state and the snapshot falls within ``HORIZON_H`` hours before that
   end. A window inherits the majority vote of its snapshots.
3. The train/test split is **temporal** (by job end date), never random.
4. Model selection uses a **temporal holdout** carved out of train, and
   robustness is checked with **purged, expanding time-series CV**. Shuffled or
   merely grouped CV is never used to select anything - Session 2 measured them
   inflating average precision by 5.7x and 2.7x over the truth.
5. Every window carries its own **end timestamp**, so any split, fold or embargo
   can be expressed in time rather than in row order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths - self-contained under pipeline_v2/, except the read-only source parquet
# --------------------------------------------------------------------------- #
PV2 = Path(__file__).resolve().parent
ROOT = PV2.parent
DATASET = ROOT / "dataset" / "prom_slurm_joined" / "prom_slurm_joined.parquet"

ARTIFACTS = PV2 / "artifacts"
RESULTS = PV2 / "results"
FIGURES = RESULTS / "figures"
DOCS = PV2 / "docs"
SCAFFOLD = ARTIFACTS / "scaffold.parquet"      # per-snapshot key columns (nb 01)
JOBTABLE = ARTIFACTS / "job_table.parquet"     # one row per job (nb 01)

for _d in (ARTIFACTS, RESULTS, FIGURES, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------------------------------- #
# Dataset facts - verified in notebook 01, not assumed
# --------------------------------------------------------------------------- #
SNAP_SECONDS = 30.0     # telemetry cadence. Session 1 documented 15 s; it is 30 s.
GAP_TOL_S = 60.0        # a window is rejected if any internal gap exceeds this

# --------------------------------------------------------------------------- #
# Primary pipeline configuration
# --------------------------------------------------------------------------- #
WINDOW = 40             # held at 40 for the feature work so the numbers stay
                        # comparable with Sessions 1-2; varied in notebook 05
HORIZON_H = 2.0         # hours before job end that count as "pre-failure"
TRAIN_FRACTION = 0.8    # first 80 % of jobs by end date -> train
VAL_FRACTION = 0.15     # last 15 % of *train* jobs by end date -> temporal holdout
STRIDE_TRAIN = 5        # train windows are subsampled: 39/40 overlap is redundant
STRIDE_TEST = 1         # test keeps every window, so the test population (and the
                        # PR-AUC baseline) stays comparable with Sessions 1-2

FAILURE_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
STATE_CODES = {"COMPLETED": 0, "TIMEOUT": 1, "FAILED": 2, "OUT_OF_MEMORY": 3,
               "NODE_FAIL": 4}
CODE_TO_STATE = {v: k for k, v in STATE_CODES.items()}
# "OOM" is the OUT_OF_MEMORY *job state*, never the machine running out of RAM.
STATE_DISPLAY = {"COMPLETED": "COMPLETED", "TIMEOUT": "TIMEOUT", "FAILED": "FAILED",
                 "OUT_OF_MEMORY": "OOM", "NODE_FAIL": "NODE_FAIL"}
REPORT_STATES = ["TIMEOUT", "FAILED", "OUT_OF_MEMORY", "NODE_FAIL"]
# Failures that plausibly reflect *node health*. TIMEOUT is usually a user
# under-estimating walltime; notebook 05 tests dropping it from the positives.
HARDWARE_FAILURES = ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL")

SCAFFOLD_COLS = ["slurm_id", "node", "timestamp", "state", "end_date"]

# --------------------------------------------------------------------------- #
# Metric registry
# --------------------------------------------------------------------------- #
# Each metric is (name, kind, source). Kinds:
#   "raw"    -> the parquet column as-is
#   "rate"   -> per-second first difference of a cumulative counter; negatives
#               (counter resets) clipped to 0, first snapshot back-filled
#   "spread" -> difference of two columns (max - min across the node's GPUs)
#
# Group V1 reproduces Session 1's eight metrics *exactly*, including the three
# that are raw cumulative counters. That is deliberate: rung A of the ablation
# has to be a faithful replica before any later change can be credited.
V1 = [
    ("mem_active",       "raw", "node_memory_Active_bytes"),
    ("disk_written_cum", "raw", "node_disk_written_bytes_total_sum"),
    ("tcp_inerrs_cum",   "raw", "node_netstat_Tcp_InErrs"),
    ("forks_cum",        "raw", "node_forks_total"),
    ("gpu_temp_mean",    "raw", "nvidia_gpu_temperature_celsius_mean"),
    ("gpu_power_mean",   "raw", "nvidia_gpu_power_usage_milliwatts_mean"),
    ("gpu_mem_used",     "raw", "nvidia_gpu_memory_used_bytes_sum"),
    ("gpu_fan_mean",     "raw", "nvidia_gpu_fanspeed_percent_mean"),
]

# Group RATE: those same counters as *rates*, plus the other counters in the
# dump. A cumulative counter summarised inside a 20-minute window mostly encodes
# how long the node has been up; its rate encodes what the node is doing now.
RATE = [
    ("disk_written_rate", "rate", "node_disk_written_bytes_total_sum"),
    ("tcp_inerrs_rate",   "rate", "node_netstat_Tcp_InErrs"),
    ("forks_rate",        "rate", "node_forks_total"),
    ("ctxsw_rate",        "rate", "node_context_switches_total"),
    ("intr_rate",         "rate", "node_intr_total"),
    ("disk_read_rate",    "rate", "node_disk_read_bytes_total_sum"),
    ("disk_wcompl_rate",  "rate", "node_disk_writes_completed_total_sum"),
    ("net_tx_rate",       "rate", "node_network_transmit_bytes_total_sum"),
    ("net_rx_rate",       "rate", "node_network_receive_bytes_total_sum"),
    ("net_rxdrop_rate",   "rate", "node_network_receive_drop_total_sum"),
    ("tcp_retrans_rate",  "rate", "node_netstat_Tcp_RetransSegs"),
    ("icmp_inerr_rate",   "rate", "node_netstat_Icmp_InErrors"),
    ("rapl_power",        "raw",  "node_rapl_package_power_sum"),
]

# Group GPU: GPU telemetry Session 1 left on the table. Duty cycle is
# utilisation (a stalled job shows up as a dead GPU); the max-min spreads expose
# one sick GPU inside an otherwise healthy node.
GPU = [
    ("gpu_duty_mean",    "raw",    "nvidia_gpu_duty_cycle_mean"),
    ("gpu_duty_min",     "raw",    "nvidia_gpu_duty_cycle_min"),
    ("gpu_temp_max",     "raw",    "nvidia_gpu_temperature_celsius_max"),
    ("gpu_temp_spread",  "spread", ("nvidia_gpu_temperature_celsius_max",
                                    "nvidia_gpu_temperature_celsius_min")),
    ("gpu_power_max",    "raw",    "nvidia_gpu_power_usage_milliwatts_max"),
    ("gpu_power_spread", "spread", ("nvidia_gpu_power_usage_milliwatts_max",
                                    "nvidia_gpu_power_usage_milliwatts_min")),
    ("gpu_fan_max",      "raw",    "nvidia_gpu_fanspeed_percent_max"),
]

# Group NODE: host-side health. MemFree/Dirty speak to the OUT_OF_MEMORY state,
# procs_blocked to I/O stalls, the temperatures to thermal trouble.
NODE = [
    ("mem_free",       "raw", "node_memory_MemFree_bytes"),
    ("mem_dirty",      "raw", "node_memory_Dirty_bytes"),
    ("load1",          "raw", "node_load1"),
    ("load15",         "raw", "node_load15"),
    ("procs_running",  "raw", "node_procs_running"),
    ("procs_blocked",  "raw", "node_procs_blocked"),
    ("hwmon_temp_max", "raw", "node_hwmon_temp_celsius_max"),
    ("thermal_max",    "raw", "node_thermal_zone_temp_max"),
    ("node_power",     "raw", "node_power_usage"),
    ("fs_avail",       "raw", "node_filesystem_avail_bytes_sum"),
    ("disk_io_now",    "raw", "node_disk_io_now_sum"),
]

METRICS = V1 + RATE + GPU + NODE
METRIC_NAMES = [m[0] for m in METRICS]
METRIC_GROUP = {}
for _grp, _lst in (("v1", V1), ("rate", RATE), ("gpu", GPU), ("node", NODE)):
    for _m in _lst:
        METRIC_GROUP[_m[0]] = _grp

# --------------------------------------------------------------------------- #
# Within-window aggregations
# --------------------------------------------------------------------------- #
# Session 1 used min/max/mean/std only, which throw away the *trajectory*: a
# window where memory climbs steadily and one where it is merely noisy get
# identical summaries. last/delta/slope restore direction and current state.
AGGS = ["min", "max", "mean", "std", "last", "delta", "slope"]
AGGS_V1 = ["min", "max", "mean", "std"]
# Aggregations that are pure scale (no offset) - needed by `node_normalise`.
SCALE_ONLY_AGGS = {"std", "delta", "slope"}


def feature_names(metrics=None, aggs=None) -> list[str]:
    """Column names of the built matrix, in build order (metric-major)."""
    metrics = METRIC_NAMES if metrics is None else metrics
    aggs = AGGS if aggs is None else aggs
    return [f"{m}__{a}" for m in metrics for a in aggs]


def source_columns(metrics=None) -> list[str]:
    """Distinct parquet columns needed to compute the given metrics."""
    metrics = METRICS if metrics is None else metrics
    cols: list[str] = []
    for _name, kind, src in metrics:
        for c in ((src,) if isinstance(src, str) else src):
            if c not in cols:
                cols.append(c)
    return cols


def slope_weights(window: int = WINDOW, snap_seconds: float = SNAP_SECONDS):
    """Weights w such that ``w @ x`` is the OLS slope of x against time, per sec.

    Valid because the contiguity guard guarantees an even time grid inside every
    accepted window, so the design matrix is identical for all of them.
    """
    i = np.arange(window, dtype=np.float64)
    c = i - i.mean()
    return c / (c @ c) / snap_seconds


# --------------------------------------------------------------------------- #
# Matrix storage: raw .bin + a JSON sidecar
# --------------------------------------------------------------------------- #
# The number of surviving windows is known exactly before the build (pass 0
# counts them from timestamps alone), but writing plain binary keeps the builder
# a simple append loop and lets every consumer memory-map the result read-only -
# which is what keeps this pipeline inside a 16 GB laptop.

def save_meta(path: Path, meta: dict) -> None:
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def open_matrix(path: Path, mode: str = "r"):
    """Memory-map a built matrix. Returns (array, meta)."""
    meta = load_meta(path)
    arr = np.memmap(path, dtype=np.dtype(meta["dtype"]), mode=mode,
                    shape=tuple(meta["shape"]))
    return arr, meta


def dataset_dir(tag: str) -> Path:
    """Where one built dataset variant lives (``primary``, ``w20``, ...)."""
    d = ARTIFACTS / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_dataset(tag: str, split: str):
    """Load one split of a built dataset: X memory-mapped, everything else in RAM."""
    d = dataset_dir(tag)
    X, meta = open_matrix(d / f"X_{split}.bin")
    out = {"X": X, "features": meta["features"], "meta": meta}
    for name in ("y", "job_id", "node_id", "state", "t_end", "ttf_end"):
        p = d / f"{name}_{split}.npy"
        if p.exists():
            out[name] = np.load(p)
    return out


# --------------------------------------------------------------------------- #
# Column selection helpers (the built matrix is wide; models use subsets)
# --------------------------------------------------------------------------- #
def column_index(features: list[str], metrics=None, aggs=None,
                 groups=None, names=None) -> np.ndarray:
    """Indices of the columns matching a metric / aggregation / group filter."""
    idx = []
    for i, f in enumerate(features):
        m, a = f.split("__")
        if names is not None and f not in names:
            continue
        if metrics is not None and m not in metrics:
            continue
        if aggs is not None and a not in aggs:
            continue
        if groups is not None and METRIC_GROUP.get(m) not in groups:
            continue
        idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def take_columns(X, cols, rows=None, chunk: int = 200_000):
    """Materialise a column (and optionally row) subset of a memory-mapped matrix.

    Chunked on purpose: slicing rows first and columns second would briefly hold
    all 273 columns of every selected row in RAM, which is most of the budget on
    this machine.
    """
    cols = np.asarray(cols, dtype=np.int64)
    if rows is None:
        idx, n = None, X.shape[0]
    else:
        rows = np.asarray(rows)
        idx = np.flatnonzero(rows) if rows.dtype == bool else rows
        n = len(idx)
    out = np.empty((n, len(cols)), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = (X[s:e][:, cols] if idx is None else X[idx[s:e]][:, cols])
    return out


def _norm_plan(features_sub: list[str], baseline: dict):
    """Pre-compute the per-column metric index and offset/scale rule."""
    m_index = {name: i for i, name in enumerate(baseline["metrics"])}
    cols_m = np.array([m_index[f.split("__")[0]] for f in features_sub])
    scale_only = np.array([f.split("__")[1] in SCALE_ONLY_AGGS for f in features_sub])
    return cols_m, scale_only


def _norm_block(block, node_id, baseline, plan):
    """Node-standardise one chunk of already-materialised feature columns."""
    cols_m, scale_only = plan
    mu = baseline["mean"][node_id][:, cols_m]
    sd = baseline["std"][node_id][:, cols_m]
    b = block.astype(np.float64)
    return np.where(scale_only, b / sd, (b - mu) / sd).astype(np.float32)


def node_normalise(Xsub, features_sub: list[str], node_id, baseline: dict,
                   chunk: int = 200_000):
    """Express every feature as a deviation from that node's own normal.

    80 C means different things on different GPUs; what carries health
    information is the departure from the node's own baseline. Because the
    snapshot-level transform ``z = (x - m_node) / s_node`` is affine and the
    window aggregations are homogeneous, the normalised aggregates follow from
    the raw ones with no second pass over the parquet:

        min/max/mean/last -> (v - m) / s        (offset and scale)
        std/delta/slope   ->  v / s             (scale only)

    `baseline` holds per-node mean/std of each *metric*, computed from training
    snapshots only.
    """
    plan = _norm_plan(features_sub, baseline)
    out = np.empty_like(Xsub)
    for s in range(0, Xsub.shape[0], chunk):
        e = min(s + chunk, Xsub.shape[0])
        out[s:e] = _norm_block(Xsub[s:e], node_id[s:e], baseline, plan)
    return out


# =========================================================================== #
# THE BUILD
# =========================================================================== #
# Two reads of the parquet, both streaming partition by partition:
#   * `build_scaffold` takes the five key columns and caches them, so every
#     audit / labelling / split question can be answered without touching the
#     10 GB again;
#   * `build_dataset` takes the key columns plus the metric columns and turns
#     each job into windows, appending straight to disk.
# Peak RAM stays around one partition (a few hundred MB) rather than the whole
# frame, which is what makes this run on a machine with ~4 GB free.

def _partition_files() -> list[Path]:
    return sorted(p for p in DATASET.glob("*.parquet"))


def build_scaffold(force: bool = False, verbose: bool = True):
    """Cache slurm_id / node / timestamp / state / end_date for every GPU snapshot.

    Filters applied at the Arrow level, matching the project's standing
    definition of the population: ``gpu_node == 1`` and ``state != CANCELLED``
    (a cancelled job says nothing about node health).
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if SCAFFOLD.exists() and JOBTABLE.exists() and not force:
        if verbose:
            print(f"scaffold cached -> {SCAFFOLD.name}, {JOBTABLE.name}")
        return pd.read_parquet(SCAFFOLD), pd.read_parquet(JOBTABLE)

    files = _partition_files()
    frames = []
    for i, p in enumerate(files):
        t = pq.read_table(p, columns=SCAFFOLD_COLS + ["gpu_node"])
        t = t.filter((pc.field("gpu_node") == 1) & (pc.field("state") != "CANCELLED"))
        if t.num_rows == 0:
            continue
        df = t.drop_columns(["gpu_node"]).to_pandas()
        df["part"] = np.int16(i)
        frames.append(df)
        if verbose and (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(files)} partitions")
    sc = pd.concat(frames, ignore_index=True)
    sc["state"] = sc["state"].astype("category")
    sc["node"] = sc["node"].astype("category")
    # A job can span several nodes, and the join emits one row per (node,
    # timestamp). Sorting only by (slurm_id, timestamp) therefore *interleaves*
    # two machines inside one window - which is what Session 1 did for the 123
    # multi-node jobs. The windowing unit here is the (job, node) series.
    sc = sc.sort_values(["slurm_id", "node", "timestamp"], kind="stable").reset_index(drop=True)
    sc["series_id"] = (sc.groupby(["slurm_id", "node"], sort=False, observed=True)
                         .ngroup().astype(np.int32))

    g = sc.groupby("slurm_id", sort=True, observed=True)
    jt = pd.DataFrame({
        "n_snap": g.size(),
        "n_nodes": g["node"].nunique(),
        "node": g["node"].first(),
        "state": g["state"].first(),
        "end_date": g["end_date"].first(),
        "first_ts": g["timestamp"].min(),
        "last_ts": g["timestamp"].max(),
        "last_part": g["part"].max(),
    }).reset_index()
    jt["is_failure"] = jt["state"].isin(FAILURE_STATES)
    # Temporal split: the earliest 80 % of jobs by end date are train. Ordering is
    # made deterministic by breaking ties on slurm_id.
    order = jt.sort_values(["end_date", "slurm_id"], kind="stable").index
    n_train = int(len(jt) * TRAIN_FRACTION)
    jt["split"] = "test"
    jt.loc[order[:n_train], "split"] = "train"
    jt["split_rank"] = np.argsort(np.argsort(order.to_numpy()))

    sc.to_parquet(SCAFFOLD, index=False)
    jt.to_parquet(JOBTABLE, index=False)
    if verbose:
        print(f"scaffold: {len(sc):,} snapshots, {len(jt):,} jobs -> {SCAFFOLD.name}")
    return sc, jt


def valid_window_starts(ts_ns, window: int, stride: int, gap_tol_s: float = GAP_TOL_S):
    """Start indices of the windows we accept, on one job's sorted timestamps.

    A window is accepted only when every gap inside it is <= `gap_tol_s`. That
    rejects the handful of windows that straddle a monitoring outage - whose
    min/max/mean/slope would otherwise mix telemetry hours apart - and, just as
    usefully, it guarantees an even time grid inside every surviving window. The
    even grid is what makes `slope` a well-defined OLS slope and what lets
    notebook 05 re-derive labels for any horizon in closed form.
    """
    n = len(ts_ns)
    if n < window:
        return np.empty(0, dtype=np.int64)
    cand = np.arange(0, n - window + 1, stride, dtype=np.int64)
    if window > 1:
        dt = np.diff(ts_ns) / 1e9
        # rolling max of the window-1 gaps inside each candidate window
        gaps = np.lib.stride_tricks.sliding_window_view(dt, window - 1)
        ok = gaps[cand].max(axis=1) <= gap_tol_s
        cand = cand[ok]
    return cand


def metric_matrix(df: pd.DataFrame, metrics, ts_ns) -> np.ndarray:
    """(n_snapshots, n_metrics) float64 for one job, rates already differenced."""
    n = len(df)
    dt = np.full(n, np.nan)
    if n > 1:
        d = np.diff(ts_ns) / 1e9
        dt[1:] = np.where(d > 0, d, np.nan)
    out = np.empty((n, len(metrics)), dtype=np.float64)
    for j, (_name, kind, src) in enumerate(metrics):
        if kind == "raw":
            out[:, j] = df[src].to_numpy(dtype=np.float64)
        elif kind == "spread":
            out[:, j] = (df[src[0]].to_numpy(dtype=np.float64)
                         - df[src[1]].to_numpy(dtype=np.float64))
        elif kind == "rate":
            x = df[src].to_numpy(dtype=np.float64)
            r = np.full(n, np.nan)
            if n > 1:
                r[1:] = np.diff(x) / dt[1:]
                np.clip(r, 0.0, None, out=r)   # a negative step is a counter reset
                r[0] = r[1]
            else:
                r[0] = 0.0
            out[:, j] = r
        else:
            raise ValueError(f"unknown metric kind {kind!r}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def aggregate_windows(vals, starts, window: int, aggs, slope_w, chunk: int = 2048):
    """Summarise each accepted window -> (n_windows, n_metrics * n_aggs) float32.

    Column order is metric-major: all aggregations of metric 0, then metric 1...
    """
    n_m = vals.shape[1]
    out = np.empty((len(starts), n_m * len(aggs)), dtype=np.float32)
    sw = np.lib.stride_tricks.sliding_window_view(vals, window, axis=0)  # (S, M, W)
    for s in range(0, len(starts), chunk):
        sub = sw[starts[s:s + chunk]]            # (c, M, W) materialised copy
        parts = {}
        if "min" in aggs:
            parts["min"] = sub.min(-1)
        if "max" in aggs:
            parts["max"] = sub.max(-1)
        if "mean" in aggs:
            parts["mean"] = sub.mean(-1)
        if "std" in aggs:
            parts["std"] = sub.std(-1)          # ddof = 0, as in Session 1
        if {"last", "delta"} & set(aggs):
            last = sub[..., -1]
            parts["last"] = last
            parts["delta"] = last - sub[..., 0]
        if "slope" in aggs:
            parts["slope"] = sub @ slope_w
        stacked = np.stack([parts[a] for a in aggs], axis=2)   # (c, M, A)
        out[s:s + len(sub)] = stacked.reshape(len(sub), n_m * len(aggs))
    return out


def build_dataset(tag: str = "primary", metrics=None, window: int = WINDOW,
                  horizon_h: float = HORIZON_H, aggs=None,
                  stride_train: int = STRIDE_TRAIN, stride_test: int = STRIDE_TEST,
                  positive_states=FAILURE_STATES, force: bool = False,
                  verbose: bool = True) -> dict:
    """Stream the parquet once and write one windowed dataset variant to disk.

    Per split it writes ``X_{split}.bin`` (float32, memory-mappable) plus the
    aligned side arrays that make every later question answerable **in time**:

        y        window label (majority vote of its snapshots)
        job_id   the job the window came from     (grouping)
        node_id  the node the job ran on          (per-node normalisation)
        state    the job's final Slurm state      (per-state recall)
        t_end    timestamp of the window's LAST snapshot, int64 ns  (all CV)
        ttf_end  seconds from that snapshot to the job's end        (horizons)

    It also accumulates per-node mean/std of every metric over **training
    snapshots only**, for the node-normalisation ablation.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    metrics = METRICS if metrics is None else metrics
    aggs = AGGS if aggs is None else aggs
    d = dataset_dir(tag)
    meta_path = d / "X_train.bin"
    if (meta_path.with_suffix(".json")).exists() and not force:
        if verbose:
            print(f"dataset '{tag}' already built -> {d}")
        return load_meta(meta_path)

    _sc, jt = build_scaffold(verbose=False)
    split_of = dict(zip(jt["slurm_id"], jt["split"]))
    last_part = dict(zip(jt["slurm_id"], jt["last_part"]))
    nodes = sorted(jt["node"].astype(str).unique())
    node_code = {n: i for i, n in enumerate(nodes)}

    cols = source_columns(metrics)
    feats = feature_names([m[0] for m in metrics], aggs)
    slope_w = slope_weights(window)
    horizon_s = horizon_h * 3600.0
    pos_states = set(positive_states)
    strides = {"train": stride_train, "test": stride_test}

    files = _partition_files()
    handles = {s: open(d / f"X_{s}.bin", "wb") for s in ("train", "test")}
    aux = {s: {k: [] for k in ("y", "job_id", "node_id", "state", "t_end", "ttf_end")}
           for s in ("train", "test")}
    counts = {"train": 0, "test": 0}
    dropped_short = 0
    dropped_gap = 0
    # per-node accumulators over training snapshots (float64 sums)
    n_m = len(metrics)
    acc = {k: np.zeros((len(nodes), n_m)) for k in ("n", "s", "ss")}

    def process(jid, sub):
        # One job can span several nodes; each (job, node) series is windowed on
        # its own, never interleaved.
        split = split_of.get(jid)
        if split is None:
            return
        for _node, ser in sub.groupby("node", sort=True, observed=True):
            _process_series(jid, ser, split)

    def _process_series(jid, sub, split):
        nonlocal dropped_short, dropped_gap
        sub = sub.sort_values("timestamp", kind="stable")
        ts = sub["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        n = len(ts)
        vals = metric_matrix(sub, metrics, ts)
        nid = node_code[str(sub["node"].iloc[0])]
        if split == "train":                       # baseline stats: train only
            acc["n"][nid] += n
            acc["s"][nid] += vals.sum(0)
            acc["ss"][nid] += (vals ** 2).sum(0)
        if n < window:
            dropped_short += 1
            return
        starts = valid_window_starts(ts, window, strides[split])
        n_cand = len(range(0, n - window + 1, strides[split]))
        dropped_gap += n_cand - len(starts)
        if len(starts) == 0:
            return

        state = str(sub["state"].iloc[0])
        end_ns = sub["end_date"].iloc[0].to_datetime64().astype(np.int64)
        ttf = (end_ns - ts) / 1e9                                   # seconds to end
        if state in pos_states:
            snap_pos = ((ttf >= 0.0) & (ttf <= horizon_s)).astype(np.int32)
        else:
            snap_pos = np.zeros(n, dtype=np.int32)
        cs = np.concatenate([[0], np.cumsum(snap_pos)])
        n_pos = cs[starts + window] - cs[starts]
        y = (n_pos >= window / 2).astype(np.int8)                   # majority vote

        X = aggregate_windows(vals, starts, window, aggs, slope_w)
        handles[split].write(X.tobytes())
        end_idx = starts + window - 1
        a = aux[split]
        a["y"].append(y)
        a["job_id"].append(np.full(len(starts), jid, dtype=np.int32))
        a["node_id"].append(np.full(len(starts), nid, dtype=np.int8))
        a["state"].append(np.full(len(starts), STATE_CODES[state], dtype=np.int8))
        a["t_end"].append(ts[end_idx])
        a["ttf_end"].append(ttf[end_idx].astype(np.float32))
        counts[split] += len(starts)

    carry: dict[int, list] = {}
    t0 = pd.Timestamp.now()
    for i, p in enumerate(files):
        t = pq.read_table(p, columns=SCAFFOLD_COLS + cols + ["gpu_node"])
        t = t.filter((pc.field("gpu_node") == 1) & (pc.field("state") != "CANCELLED"))
        if t.num_rows == 0:
            continue
        df = t.drop_columns(["gpu_node"]).to_pandas()
        for jid, sub in df.groupby("slurm_id", sort=False):
            # 99 % of jobs live inside a single partition; the rest are held back
            # until their last partition has been read.
            if last_part.get(jid, -1) == i:
                if jid in carry:
                    sub = pd.concat(carry.pop(jid) + [sub], ignore_index=True)
                process(jid, sub)
            else:
                carry.setdefault(jid, []).append(sub)
        if verbose and (i + 1) % 25 == 0:
            el = (pd.Timestamp.now() - t0).total_seconds()
            print(f"  {i + 1:>3}/{len(files)} partitions | "
                  f"train {counts['train']:>9,} | test {counts['test']:>9,} | {el:5.0f}s")
    for jid, chunks in list(carry.items()):        # safety net; should be empty
        process(jid, pd.concat(chunks, ignore_index=True))
    for h in handles.values():
        h.close()

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = acc["s"] / acc["n"]
        var = acc["ss"] / acc["n"] - mean ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    gmean = np.nansum(acc["s"], 0) / np.nansum(acc["n"], 0)
    mean = np.where(np.isfinite(mean), mean, gmean)                 # node unseen in train
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)          # constant metric
    np.savez(d / "node_baseline.npz", mean=mean, std=std,
             metrics=np.array([m[0] for m in metrics]), nodes=np.array(nodes))

    meta = {"tag": tag, "window": window, "horizon_h": horizon_h, "aggs": aggs,
            "metrics": [m[0] for m in metrics], "features": feats,
            "stride_train": stride_train, "stride_test": stride_test,
            "positive_states": sorted(pos_states), "gap_tol_s": GAP_TOL_S,
            "dropped_jobs_too_short": dropped_short,
            "dropped_windows_gap": int(dropped_gap),
            "dtype": "float32", "built": pd.Timestamp.now().isoformat()}
    for s in ("train", "test"):
        for k, v in aux[s].items():
            np.save(d / f"{k}_{s}.npy", np.concatenate(v) if v else np.array([]))
        m = dict(meta)
        m["shape"] = [counts[s], len(feats)]
        m["split"] = s
        save_meta(d / f"X_{s}.bin", m)
    if verbose:
        el = (pd.Timestamp.now() - t0).total_seconds()
        print(f"built '{tag}' in {el:.0f}s | train {counts['train']:,} x {len(feats)} "
              f"| test {counts['test']:,} x {len(feats)} | "
              f"{dropped_short:,} jobs too short, {dropped_gap:,} windows spanned a gap")
    return load_meta(d / "X_train.bin")


def relabel(ttf_end, state, window: int = WINDOW, horizon_h: float = HORIZON_H,
            positive_states=FAILURE_STATES, majority: bool = True,
            snap_seconds: float = SNAP_SECONDS):
    """Recompute window labels for a different horizon or positive class.

    No rebuild required. Because the gap guard leaves every accepted window on an
    even grid, the j-th snapshot counted back from the window's end sits exactly
    ``j * snap_seconds`` further from the job's end, so the number of snapshots
    inside a horizon follows in closed form from `ttf_end` alone.

    `majority=True` reproduces the >= W/2 rule; `majority=False` is the "any
    snapshot inside the horizon" rule tested in notebook 05.
    """
    ttf = np.asarray(ttf_end, dtype=np.float64)
    horizon_s = horizon_h * 3600.0
    j_hi = np.floor((horizon_s - ttf) / snap_seconds)          # last j inside horizon
    j_lo = np.ceil(np.maximum(0.0, -ttf) / snap_seconds)       # first j with ttf >= 0
    n_pos = np.clip(j_hi - j_lo + 1.0, 0.0, float(window))
    need = window / 2.0 if majority else 1.0
    codes = np.array([STATE_CODES[s] for s in positive_states])
    return ((n_pos >= need) & np.isin(state, codes)).astype(np.int8)


# =========================================================================== #
# TIME-AWARE SPLITS  (invariant 4)
# =========================================================================== #
def temporal_holdout(t_end, job_id, val_fraction: float = VAL_FRACTION,
                     embargo_s: float | None = None):
    """Split TRAIN windows into (fit, validate) by time, purged and embargoed.

    * validation is the most recent `val_fraction` of train windows;
    * any job with a window in validation is removed from fit entirely, so the
      39/40-overlapping siblings of a validation window can never be trained on
      (this is the leak Session 2 measured at 5.7x);
    * an extra `embargo_s` of fit data just before the boundary is dropped,
      because a window's label looks up to one horizon into the future.
    """
    embargo_s = HORIZON_H * 3600.0 if embargo_s is None else embargo_s
    t_split = np.quantile(t_end, 1.0 - val_fraction)
    val = t_end >= t_split
    val_jobs = np.unique(job_id[val])
    fit = (t_end < t_split - embargo_s * 1e9) & ~np.isin(job_id, val_jobs)
    return fit, val


def purged_time_folds(t_end, job_id, n_splits: int = 5, embargo_s: float | None = None):
    """Expanding-window CV: fold k trains on everything before block k.

    Same purge and embargo as `temporal_holdout`, applied per fold. This is the
    only CV used for model-selection robustness checks in this session.
    """
    embargo_s = HORIZON_H * 3600.0 if embargo_s is None else embargo_s
    edges = np.quantile(t_end, np.linspace(0, 1, n_splits + 2)[1:])
    folds = []
    for k in range(n_splits):
        lo, hi = edges[k], edges[k + 1]
        val = (t_end >= lo) & (t_end < hi)
        if val.sum() == 0:
            continue
        val_jobs = np.unique(job_id[val])
        fit = (t_end < lo - embargo_s * 1e9) & ~np.isin(job_id, val_jobs)
        if fit.sum() == 0:
            continue
        folds.append((np.flatnonzero(fit), np.flatnonzero(val)))
    return folds


# =========================================================================== #
# MODEL PLUMBING
# =========================================================================== #
# Everything downstream trains on a *column subset* of the one wide matrix, so
# these two helpers - materialise a view, predict a memory-mapped split in
# chunks - are all the plumbing the notebooks need.

def materialise(d, cols, rows=None, node_norm: bool = False, baseline=None,
                add_node: bool = False, norm_cols=None, chunk: int = 200_000):
    """Build the model matrix for one column subset.

    Assembled chunk by chunk into a single pre-allocated array - concatenating
    finished blocks instead would momentarily double the largest object in the
    process, which this machine cannot afford.

    `node_norm` replaces the selected columns with node-standardised versions;
    `norm_cols` *appends* node-standardised copies of some columns next to the
    raw ones (the two are different hypotheses, and rungs E and G test both).
    """
    if rows is None:
        idx, n = None, d["X"].shape[0]
    else:
        rows = np.asarray(rows)
        idx = np.flatnonzero(rows) if rows.dtype == bool else rows
        n = len(idx)
    cols = np.asarray(cols, dtype=np.int64)
    norm_cols = None if norm_cols is None else np.asarray(norm_cols, dtype=np.int64)
    n_extra = (0 if norm_cols is None else len(norm_cols)) + int(add_node)
    out = np.empty((n, len(cols) + n_extra), dtype=np.float32)

    plan = (_norm_plan([d["features"][c] for c in cols], baseline) if node_norm else None)
    plan_x = (_norm_plan([d["features"][c] for c in norm_cols], baseline)
              if norm_cols is not None else None)
    node_all = d["node_id"]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        rid = np.arange(s, e) if idx is None else idx[s:e]
        block = d["X"][rid]
        nid = node_all[rid]
        sub = block[:, cols]
        out[s:e, :len(cols)] = _norm_block(sub, nid, baseline, plan) if node_norm else sub
        if norm_cols is not None:
            out[s:e, len(cols):len(cols) + len(norm_cols)] = _norm_block(
                block[:, norm_cols], nid, baseline, plan_x)
        if add_node:
            out[s:e, -1] = nid
    return out


def lgbm_params(y_fit, **over) -> dict:
    """The fixed LightGBM configuration used as the ablation workhorse.

    `metric="average_precision"` matters: Session 2 lost a whole experiment to
    early stopping that silently watched `binary_logloss` and quit at round 1.
    """
    pos = float(np.sum(y_fit == 1))
    p = {"objective": "binary", "metric": "average_precision", "first_metric_only": True,
         "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
         "scale_pos_weight": (len(y_fit) - pos) / max(pos, 1.0),
         "num_threads": 10, "seed": SEED, "verbose": -1}
    p.update(over)
    return p


def train_lgbm(X_fit, y_fit, X_val, y_val, params=None, rounds: int = 800,
               patience: int = 50, log_every: int = 0, free_inputs: bool = True):
    """Fit with early stopping on a *temporal* validation set (invariant 4)."""
    import gc
    import lightgbm as lgb
    params = lgbm_params(y_fit) if params is None else params
    dtr = lgb.Dataset(X_fit, y_fit, free_raw_data=free_inputs)
    dva = lgb.Dataset(X_val, y_val, reference=dtr, free_raw_data=free_inputs)
    cbs = [lgb.early_stopping(patience, verbose=False)]
    if log_every:
        cbs.append(lgb.log_evaluation(log_every))
    bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=cbs)
    del dtr, dva
    gc.collect()
    return bst


def cv_score(X, y, folds, params=None, rounds: int = 600, patience: int = 40):
    """Average precision of one candidate across purged expanding-window folds.

    Notebook 03 showed a single temporal holdout ranks feature sets unreliably -
    it is one period, and which columns help is period-specific. Averaging over
    several folds is the fix, and it is still strictly time-ordered: every fold
    trains on the past and validates on the next block, purged of shared jobs
    and embargoed by one horizon.
    """
    import gc
    aps, iters = [], []
    for fit_idx, val_idx in folds:
        Xf, yf = X[fit_idx], y[fit_idx]
        Xv, yv = X[val_idx], y[val_idx]
        p = lgbm_params(yf) if params is None else {**params,
                                                    **{"scale_pos_weight":
                                                       lgbm_params(yf)["scale_pos_weight"]}}
        bst = train_lgbm(Xf, yf, Xv, yv, params=p, rounds=rounds, patience=patience)
        aps.append(float(bst.best_score["valid_0"]["average_precision"]))
        iters.append(int(bst.best_iteration))
        del Xf, Xv, bst
        gc.collect()
    return {"cv_ap": float(np.mean(aps)), "cv_sd": float(np.std(aps)),
            "rounds": int(np.mean(iters)), "fold_aps": aps, "fold_rounds": iters}


def predict_split(model, d, cols, node_norm: bool = False, baseline=None,
                  add_node: bool = False, norm_cols=None, num_iteration=None,
                  chunk: int = 300_000):
    """Score a whole split without ever holding more than `chunk` rows of it."""
    n = d["X"].shape[0]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Xc = materialise(d, cols, np.arange(s, e), node_norm, baseline, add_node,
                         norm_cols, chunk)
        out[s:e] = model.predict(Xc, num_iteration=num_iteration)
        del Xc
    return out


def load_baseline(tag: str = "primary") -> dict:
    z = np.load(dataset_dir(tag) / "node_baseline.npz", allow_pickle=True)
    return {"mean": z["mean"], "std": z["std"], "metrics": [str(m) for m in z["metrics"]]}


# =========================================================================== #
# EVALUATION
# =========================================================================== #
# PR-AUC is the headline. Its random baseline is the *positive fraction of the
# evaluated set*, so a bare PR-AUC is unreadable on its own - every result here
# is reported together with `lift = PR-AUC / baseline`.

def evaluate(y, scores, states=None, threshold: float = 0.5, beta: float = 2.0) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    y = np.asarray(y).astype(np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    base = float(y.mean())
    ap = float(average_precision_score(y, scores))
    out = {"pr_auc": ap, "baseline": base, "lift": ap / base if base else np.nan,
           "roc_auc": float(roc_auc_score(y, scores)), "n": int(len(y)),
           "n_pos": int(y.sum()), "threshold": float(threshold)}
    pred = scores >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta ** 2
    out |= {"precision": prec, "recall": rec, "flagged": int(pred.sum()),
            "fbeta": ((1 + b2) * prec * rec / (b2 * prec + rec)) if (prec + rec) else 0.0}
    if scores.min() >= 0 and scores.max() <= 1:
        out["brier"] = float(brier_score_loss(y, scores))
    if states is not None:
        states = np.asarray(states)
        for s in REPORT_STATES:
            m = (states == STATE_CODES[s]) & (y == 1)
            out[f"recall_{STATE_DISPLAY[s]}"] = float(pred[m].mean()) if m.any() else np.nan
    return out


def best_threshold(y, scores, beta: float = 2.0):
    """Threshold maximising F-beta (beta=2 favours recall: a missed failure costs
    more than a needless drain)."""
    from sklearn.metrics import precision_recall_curve
    p, r, th = precision_recall_curve(y, scores)
    b2 = beta ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1 + b2) * p * r / (b2 * p + r)
    f = np.nan_to_num(f[:-1])
    i = int(np.argmax(f))
    return float(th[i]), float(f[i]), float(p[i]), float(r[i])


def job_level_alerts(scores, threshold, job_id, state, ttf_end, k_consecutive: int = 3):
    """Turn per-window scores into the operational question: which jobs get
    flagged, and how long before they die?

    This is *not* whole-job classification - the model still scores each window
    on its own. It only reads the alarm the way an operator would: a job is
    alerted the first time `k_consecutive` of its windows cross the threshold,
    and the lead time is how much of the job was still to run at that moment.
    """
    order = np.lexsort((-ttf_end, job_id))       # per job, earliest window first
    jid, st, ttf = job_id[order], state[order], ttf_end[order]
    hit = (scores[order] >= threshold).astype(np.int8)
    rows = []
    starts = np.flatnonzero(np.r_[True, jid[1:] != jid[:-1]])
    ends = np.r_[starts[1:], len(jid)]
    for s, e in zip(starts, ends):
        h = hit[s:e]
        lead = np.nan
        if len(h) >= k_consecutive:
            run = np.convolve(h, np.ones(k_consecutive, dtype=np.int8), "valid")
            w = np.flatnonzero(run >= k_consecutive)
            if len(w):
                lead = float(ttf[s + w[0] + k_consecutive - 1]) / 60.0   # minutes
        rows.append((int(jid[s]), int(st[s]), not np.isnan(lead), lead, e - s))
    return pd.DataFrame(rows, columns=["job_id", "state", "alerted", "lead_min",
                                       "n_windows"])


def style():
    """One matplotlib look for every figure in this session."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.autolayout": True,
    })


PALETTE = {"v1": "#8c8c8c", "rate": "#4c78a8", "gpu": "#f58518", "node": "#54a24b",
           "accent": "#e45756", "muted": "#b0b0b0"}

