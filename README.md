# Proactive HPC Node-Failure Prediction — TCC UNIFEI

Undergraduate honors thesis (Electronic Engineering, Universidade Federal de Itajubá).
The thesis builds an end-to-end pipeline for **proactive health monitoring of HPC compute
nodes**: predict that a node is about to fail, and feed that prediction back to Slurm as an
actionable signal (drain the node, requeue the jobs). This repository holds the **ML
component only** — everything trained and measured on the real historical dataset.

Full framing, related work and the pipeline architecture: [`docs/project_context.md`](docs/project_context.md).

Data source: the `prom_slurm_joined` table from the **SURF Lisa** academic cluster
(Chu et al., ICPADS 2024, [arXiv:2409.08949](https://arxiv.org/abs/2409.08949)) —
Prometheus node telemetry joined to Slurm job metadata, ~13.5 M rows over ~6 months.

## How to read this repository

**The folders are numbered in the order the work happened, and that is the order to read
them in.** Each one is a self-contained session that opens with a question, and each one's
`README.md` states the answer up front. Later sessions do not fix earlier ones; they
inherit their conclusions and test the next claim, so the numbering is an argument, not
just a filing scheme.

| # | When | Folder | The question | The answer |
|---|---|---|---|---|
| 00 | Apr – Jul | [`00_exploration/`](00_exploration/) | What is in these tables? | Column survey of the three raw tables, first look at a job's telemetry, a first HGB model. |
| 01 | Aug 6–7 | [`01_benchmark/`](01_benchmark/README.md) | Which classifier is best on a shared windowed feature set? | ~15 models ranked by PR-AUC on one frozen 32-feature matrix. Evidence-based model choice, and the CV-leakage finding that motivates session 02. |
| 02 | Aug 20–21 | [`02_cv_strategies/`](02_cv_strategies/docs/README.md) | Was that ranking honest? | Leakage-free CV, the exact PR-AUC baseline, honest tuning, operating points, calibration. |
| 03 | Aug 25–28 | [`03_pipeline_v2/`](03_pipeline_v2/README.md) | What if the pipeline is rebuilt from scratch? | A clean rebuild that imports nothing from 01/02. Reaches test **AP ≈ 0.125** (~3.1× baseline) — the number every later session is measured against. |
| 04 | Aug 30 – Sep 2 | [`04_eda/`](04_eda/README.md) | What is this dataset actually good for? | Foundational analysis that runs *before* any model: every number is a property of the data. Names co-tenancy, the TIMEOUT structure, and the horizon as the suspects. |
| 05 | Sep 3 | [`05_experiments/`](05_experiments/README.md) | Does node sharing explain the plateau? | **No.** Pre-registered test: the label ambiguity is real and exactly as large as `04` measured, but the ceiling it imposes (`AP ≈ 0.90`) sits far above where the models are. |
| 06 | Sep 3 | [`06_headroom/`](06_headroom/README.md) | Where *is* the missing performance? | Effective sample size is **jobs, not windows**. A 30-min horizon on seven columns of job context and **no telemetry** reaches job-level **AP = 0.5017**, 96 % precision at 50 flags — a walltime-exhaustion detector, and that distinction is the finding. Separately: memory drifts measurably before an OOM kill. |
| 07 | Sep 4 | [`07_allnodes/`](07_allnodes/README.md) | What if the failing-job shortage is removed? | Rebuilt over all 338 nodes: **8.5× more failing jobs, and the telemetry model's score does not move.** What it buys is one detector per failure mode. |

## Layout

```
dataset/            raw source data, immutable, gitignored (26 GB)
  prom_slurm_joined/  the 10 GB joined table every session reads
  node_hardware_info.parquet
  _unused/            prom_table_cleaned/, slurm_table_cleaned/ — surveyed in 00, unused since
docs/
  project_context.md  what the thesis is, the full pipeline, related work
  references/         TCC_Dados_ML.pdf (data-pipeline spec), Ignacy_Master_Thesis-5.pdf (Skrzeczek 2024)
00_exploration/ .. 07_allnodes/   one folder per session, in order
```

Sessions **01–03** are code-first (an importable package or module plus notebooks);
sessions **04–07** are notebook-first. Both shapes follow the same rules:

- `README.md` — the question, the answer, one row per notebook with its runtime.
- notebooks are numbered in run order, and each opens with a **pre-registration** cell (04+).
- `data/` — derived data, **gitignored**, rebuilt by re-running the session.
- `figures/`, `results/` — committed; these are what the thesis cites.
- `docs/` — session write-ups (01–03).

## Running it

```bash
pip install -r requirements.txt
```

| Session | How to run |
|---|---|
| 01 | `cd 01_benchmark && python -m benchmark.data_pipeline && python -m benchmark.run_pass1` |
| 02 | `cd 02_cv_strategies && python -m cv_strategies.run_all` |
| 03 | open `03_pipeline_v2/notebooks/*.ipynb` in order (they `import pv2` from the session folder) |
| 04–07 | open the session's notebooks in order, **with the session folder as the working directory** |

Sessions 04–07 locate the project root by walking up until they find `dataset/`, and write
`data/` and `figures/` relative to the working directory — so run them from inside their own
folder. `05` and `06` additionally read `04_eda/data/`, so run `04_eda` first.

## Conventions that hold across every session

- **Temporal split by job end date.** Never random — a random split leaks the future.
- **Prediction is per window, never per job**; a window takes the majority label of its snapshots.
- **PR-AUC / average precision is primary.** ROC-AUC is reported but is misleading here;
  positives are rare. Per-failure-state recall is always broken out, because global metrics
  hide that some failure modes are easy and others are impossible from telemetry.
- **"OOM" means the `OUT_OF_MEMORY` *job state*** — not machine RAM exhaustion.
- `random_state = 42`.
