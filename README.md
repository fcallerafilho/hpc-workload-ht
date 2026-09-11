# Proactive HPC Node-Failure Prediction — TCC UNIFEI

Undergraduate honors thesis (Electronic Engineering, Universidade Federal de Itajubá).
The thesis builds an end-to-end pipeline for **proactive health monitoring of HPC compute
nodes**: predict that a node is about to fail, and feed that prediction back to Slurm as an
actionable signal (drain the node, requeue the jobs). This repository holds the **ML
component only** — everything trained and measured on the real historical dataset.

Full framing, related work and the pipeline architecture: [`docs/project_context.md`](docs/project_context.md).

Data sources:

* **SURF Lisa** (sessions 00–07) — the `prom_slurm_joined` table (Chu et al., ICPADS 2024,
  [arXiv:2409.08949](https://arxiv.org/abs/2409.08949)): Prometheus node telemetry joined
  to Slurm job metadata, 93.9 M rows over ~6 months. Node-shared telemetry.
* **PM100 / Marconi100** (session 08) — CINECA job power traces (Antici et al., SC'23
  workshops, [Zenodo 10127767](https://doi.org/10.5281/zenodo.10127767), CC-BY-4.0):
  231,238 **exclusively-allocated** jobs with per-job power at 20 s. Added to test the
  one confound Lisa cannot: telemetry that belongs to the job.

