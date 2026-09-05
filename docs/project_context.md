# Project Context — Proactive HPC Node Health Monitoring with Machine Learning

## What this project is

This is an undergraduate honors thesis (TCC) at UNIFEI (Universidade Federal de Itajubá), in Electronic Engineering. The work builds an end-to-end pipeline for **proactive health monitoring of compute nodes in an HPC (High-Performance Computing) cluster**, using machine learning to predict node failures before they impact running jobs and automatically triggering remediation through the Slurm workload manager.

The core contribution is architectural: **closing the full loop from prediction to active remediation**. Related work either stops at offline prediction (Mohammed et al. 2019, Cai et al. 2025) or uses Slurm only as a source of failure labels (Skrzeczek 2024, Utrecht MSc). This project is the first to feed the prediction back to the scheduler as an actionable signal.

## The full pipeline

```
metrics collection  →  time-series database  →  ML classifier  →  Slurm prolog/epilog  →  node drain + job requeue
```

1. **Metrics collection** — Prometheus-style node exporters emit CPU, memory, network, disk, and GPU metrics per node at a fixed sampling interval.
2. **Time-series storage** — snapshots persisted in a database (SQLite in the simulation).
3. **ML classifier** — trained offline on labeled historical data; at inference time, receives a sliding window of recent snapshots per running job and outputs the probability that the node is about to fail.
4. **Slurm integration** — when the probability crosses a threshold, a Slurm prolog/epilog script drains the node (removing it from the scheduling pool) and requeues affected jobs onto healthy nodes.

## Simulated environment (important)

The full pipeline is validated in a **fully simulated local environment**. There are no real GPUs, no real VMs, no NVML instrumentation, and no real Slurm cluster. Mock processes stand in for the node daemons and produce synthetic snapshots at the correct schema and cadence. The ML model is trained offline on the real dataset (SURF Lisa), saved as an artifact, and loaded into the local simulation for the end-to-end demonstration.

This benchmark task focuses on the **ML component only** — training and evaluating classifiers on the real historical dataset. The simulation environment is out of scope for this benchmark.

## The ML task

- **Input**: a sliding window of 40 consecutive snapshots (15 seconds each = 10-minute window) per job, aggregated as min/max/mean/std over 8 selected metrics = 32 features per window.
- **Label**: `1` if the window falls within the 2 hours before the end of a job that ended in failure (`TIMEOUT`, `FAILED`, `OOM`, `NODE_FAIL`); `0` otherwise.
- **Objective**: predict, from the current window of metrics, whether a node failure is imminent, so the scheduler can drain the node preemptively.
- **Challenge**: strong class imbalance (positives are rare) and heterogeneous failure modes (some failures are gradual hardware degradation; others are instant faults with no observable pre-failure signature).

The primary evaluation metric is **PR-AUC** (average precision), because ROC-AUC is misleading on imbalanced datasets. Per-state recall (`TIMEOUT` / `FAILED` / `OOM` / `NODE_FAIL`) is tracked separately because global metrics can hide that some failure types are easy and others are impossible with the current feature set.

## Dataset

The training data is the **`prom_slurm_joined` table from the SURF Lisa academic HPC cluster**, published by Chu et al. (ICPADS 2024, arXiv:2409.08949). It contains ~13.5 million rows joining Prometheus node metrics with Slurm job metadata, spanning about 6 months of production workload.

Relevant details:
- 100 columns total: ~80 node metrics, ~10 GPU metrics, ~10 Slurm job metadata fields.
- Subset used for this project: **GPU nodes only** (`gpu_node = 1`), which reduces the working set to ~11.1 M snapshots across ~23.6 k jobs.
- Job states retained: `COMPLETED` (majority), `TIMEOUT`, `FAILED`, `OOM`, `NODE_FAIL`. `CANCELLED` is excluded (user-driven, not a health signal).

## Where the benchmark fits

The ML component has gone through several rounds of iteration. The current model is a LightGBM classifier trained on the 32-feature windowed representation, with PR-AUC ~0.088 on a temporally-held-out test set. Multiple architectural choices have already been validated (temporal split by job `end_date` to prevent leakage, majority-vote window labels, inclusion of TIMEOUT in the positive class for training volume, etc.).

The purpose of this benchmark is to **run a systematic comparison across a broader set of classifiers** on the same feature set, so the final thesis can report an evidence-based model choice rather than relying on the three models tested so far. The technical specification for the benchmark itself (models, hyperparameters, protocol, output format) is in `benchmark_models.md`.

## Key references

- **Chu et al. 2024** (ICPADS, arXiv:2409.08949) — source dataset paper.
- **Skrzeczek 2024** (Utrecht MSc thesis) — feature selection (Chapters 5–6) used as the starting point for the 8 selected metrics.
- **Cai et al. 2025** — sliding-window formulation and PR-AUC as the primary metric for this kind of imbalanced HPC failure prediction task.
- **Mohammed et al. 2019** — earlier related work on ML for HPC failure prediction.
- **Behera et al. (HPDC 2020)** — inspiration for the simulation architecture.
- **Sîrbu & Babaoglu 2016** (arXiv:1606.04456) — counterfactual evaluation methodology used elsewhere in the thesis.
- **Kokolis et al. (Meta HPCA 2025, arXiv:2410.21680)** — industrial motivation and validation of the choice to drain nodes as the remediation action.
