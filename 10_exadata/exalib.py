"""Shared loaders for session 10 — M100 ExaData telemetry joined to the PM100 job table.

The join is direct: PM100's `nodes` are integer node ids 0..979 and ExaData's `node`
column uses the same namespace, both in UTC (verified in 09_screening).

Every ExaData metric is one Parquet folder with three columns:
    timestamp  datetime64[ms, UTC]   value  int32/float   node  string "0".."979"

`load_channel` returns it as {node_id: (ts_ms sorted, value)} so that a job's window is a
pair of searchsorted calls.  `grid_series` reduces a job to one row per GRID seconds,
averaged over whatever nodes the job held — jobs here are exclusively allocated, so that
average belongs to the job and to nothing else.
"""
from __future__ import annotations

import gc
import glob
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M100 = os.path.join(ROOT, "dataset", "m100")
PM100 = os.path.join(ROOT, "dataset", "pm100", "job_table.parquet")

GRID = 60          # seconds per row of the per-job series
SEED = 42

# channel -> (plugin, metric).  Grouped by the spec metric each one stands in for.
CHANNELS = {
    # spec 5 — GPU temperature (the mechanism the thesis names for hardware failure)
    "gpu0_temp":  ("ipmi_pub", "gpu0_core_temp"),
    "gpu1_temp":  ("ipmi_pub", "gpu1_core_temp"),
    "gpu3_temp":  ("ipmi_pub", "gpu3_core_temp"),
    "gpu4_temp":  ("ipmi_pub", "gpu4_core_temp"),
    "gpu0_mtemp": ("ipmi_pub", "gpu0_mem_temp"),
    "gpu4_mtemp": ("ipmi_pub", "gpu4_mem_temp"),
    # spec 6 — GPU power (the 507 W residual session 08 could only infer)
    "gpu0_pow":   ("ipmi_pub", "gv100card0"),
    "gpu3_pow":   ("ipmi_pub", "gv100card3"),
    # spec 8 — fan response, and the thermal context
    "fan":        ("ipmi_pub", "fan0_0"),
    "ambient":    ("ipmi_pub", "ambient"),
    # PM100's own channel, kept as the bridge to session 08
    "node_pow":   ("ipmi_pub", "total_power"),
    # spec 1 — memory occupancy (mem_active is derived from these four)
    "mem_free":   ("ganglia_pub", "mem_free"),
    "mem_cached": ("ganglia_pub", "mem_cached"),
    "mem_buf":    ("ganglia_pub", "mem_buffers"),
    "mem_total":  ("ganglia_pub", "mem_total"),
    # spec 2 — swap / disk pressure;  spec 4 — process counts
    "swap_free":  ("ganglia_pub", "swap_free"),
    "proc_run":   ("ganglia_pub", "proc_run"),
    "cpu_wio":    ("ganglia_pub", "cpu_wio"),
    "load_one":   ("ganglia_pub", "load_one"),
}

# derived after the grid series are built
DERIVED = ["mem_active", "gpu_temp_max", "gpu_pow"]

FAILURE_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
HARDWARE_STATES = ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL")


def months() -> list[str]:
    return sorted(os.path.basename(p).split("=")[1]
                  for p in glob.glob(os.path.join(M100, "year_month=*")))


def channel_files(name: str) -> list[str]:
    plugin, metric = CHANNELS[name]
    out = []
    for m in months():
        out += sorted(glob.glob(os.path.join(
            M100, f"year_month={m}", f"plugin={plugin}", f"metric={metric}", "*.parquet")))
    return out


def _load_file(path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """One month of one metric as {node_id: (timestamps ms ascending, values)}."""
    t = pq.read_table(path, columns=["timestamp", "value", "node"])
    node = pc.cast(t["node"], pa.int16()).to_numpy(zero_copy_only=False)
    ts = t["timestamp"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    val = t["value"].cast(pa.float32()).to_numpy(zero_copy_only=False)
    del t
    order = np.argsort(node, kind="stable")      # stable => time order kept inside a node
    node, ts, val = node[order], ts[order], val[order]
    del order
    uniq, starts = np.unique(node, return_index=True)
    ends = np.append(starts[1:], len(node))
    out = {}
    for u, a, b in zip(uniq, starts, ends):
        t_, v_ = ts[a:b], val[a:b]
        if len(t_) > 1 and not np.all(np.diff(t_) >= 0):
            o = np.argsort(t_, kind="stable")
            t_, v_ = t_[o], v_[o]
        out[int(u)] = (t_, v_)
    return out


def build_series(name: str, jobs: pd.DataFrame, grid: int = GRID):
    """Per-job series for one channel, one row per `grid` s, averaged over the job's nodes.

    Accumulated **one month at a time**: six months of a 20 s channel is ~600 M rows and
    will not fit in memory at once, so each file contributes to the accumulators and is
    then released.  Returns (flat values, offsets) with NaN where nothing was sampled.
    """
    step = grid * 1000
    t0 = jobs["t0"].values
    t1 = jobs["t1"].values
    nodes = jobs["nodes"].values
    Ks = ((t1 - t0) // step).astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(Ks)]).astype(np.int64)
    acc = np.zeros(offs[-1]); cnt = np.zeros(offs[-1])

    for f in channel_files(name):
        store = _load_file(f)
        if not store:
            continue
        lo = min(t_[0] for t_, _ in store.values())
        hi = max(t_[-1] for t_, _ in store.values())
        for i in np.where((t1 > lo) & (t0 < hi) & (Ks > 0))[0]:
            K, o, s, e = Ks[i], offs[i], t0[i], t1[i]
            for nd in np.atleast_1d(nodes[i]):
                g = store.get(int(nd))
                if g is None:
                    continue
                t_, v_ = g
                i0, i1 = np.searchsorted(t_, s), np.searchsorted(t_, e)
                if i1 <= i0:
                    continue
                idx = (t_[i0:i1] - s) // step
                m = (idx >= 0) & (idx < K)
                if not m.any():
                    continue
                idx = idx[m].astype(np.int64)
                acc[o:o + K] += np.bincount(idx, weights=v_[i0:i1][m].astype(np.float64),
                                            minlength=K)
                cnt[o:o + K] += np.bincount(idx, minlength=K)
        del store
        gc.collect()
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32), offs


def load_jobs(t_start: str, t_end: str, min_runtime_s: int = 1200) -> pd.DataFrame:
    """PM100 jobs that finished inside the window we hold telemetry for.

    PM100 is the exclusively-allocated subset of M100, which is what makes the telemetry
    attributable.  CANCELLED is dropped: it is a user action, not an outcome.
    """
    cols = ["job_id", "nodes", "num_nodes_alloc", "num_cores_alloc", "num_gpus_alloc",
            "start_time", "end_time", "job_state", "run_time", "time_limit", "user_id",
            "partition", "qos", "mem_req", "num_tasks", "cores_per_task", "priority"]
    J = pd.read_parquet(PM100, columns=cols)
    J = J[(J.start_time >= t_start) & (J.end_time <= t_end)
          & (J.job_state != "CANCELLED")].copy()
    # Slurm's run_time and (end - start) disagree for some jobs; the telemetry window is
    # built from the timestamps, so the wall clock is what the cohort rule must use.
    J["wall"] = (J.end_time - J.start_time).dt.total_seconds().astype("int64")
    J["drift_s"] = (J.wall - J.run_time).abs()
    J = J[J.wall >= min_runtime_s].copy()
    # ExaData timestamps are ms since epoch.  PM100's are datetime64[us], so go through an
    # explicit ms cast rather than dividing an int64 whose unit depends on the dtype.
    J["t0"] = J.start_time.values.astype("datetime64[ms]").astype("int64")
    J["t1"] = J.end_time.values.astype("datetime64[ms]").astype("int64")
    assert J.t0.min() > 1_500_000_000_000, "t0 is not milliseconds since epoch"
    assert ((J.t1 - J.t0) // 1000 == J.wall).all(), "t0/t1 disagree with the wall clock"
    J["y"] = J.job_state.isin(FAILURE_STATES).astype(int)
    return J.sort_values("end_time").reset_index(drop=True)


# ---------------------------------------------------------------- window features
AGGS = ["mean", "std", "min", "max", "slope", "delta", "last"]


def window_feats(a: np.ndarray) -> np.ndarray:
    """Seven summaries of one channel over one window; NaN when the window is empty."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.full(len(AGGS), np.nan, np.float32)
    if v.size == 1:
        return np.array([v[0], 0.0, v[0], v[0], 0.0, 0.0, v[0]], np.float32)
    x = np.arange(v.size, dtype=np.float64)
    xm, vm = x.mean(), v.mean()
    den = ((x - xm) ** 2).sum()
    slope = float(((x - xm) * (v - vm)).sum() / den) if den > 0 else 0.0
    return np.array([vm, v.std(), v.min(), v.max(), slope, v[-1] - v[0], v[-1]], np.float32)


def strat_auc(x, is_pos, is_neg, strata, min_n=5):
    """Mann-Whitney AUC pooled over strata, weighted by n_pos * n_neg.

    Pooling within strata is how a confounder (user, job size) is held fixed: a separation
    that only exists between strata contributes nothing.
    """
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    num = den = 0.0
    for s in np.unique(np.asarray(strata)[ok]):
        m = ok & (np.asarray(strata) == s)
        p, n = x[m & is_pos], x[m & is_neg]
        if len(p) < min_n or len(n) < min_n:
            continue
        r = pd.Series(np.concatenate([p, n])).rank().values
        a = (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
        w = len(p) * len(n)
        num += a * w
        den += w
    return num / den if den else np.nan
