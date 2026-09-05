# eda/ — What is in this dataset, and what is it good for?

Foundational analysis for a TCC on proactive failure prediction for GPU nodes on
an HPC cluster. It runs **before** any pipeline or model, and every number it
reports is a property of the data rather than of a modelling choice.

The source is the `prom_slurm_joined` table from the SURF Lisa academic cluster
(Chu et al., ICPADS 2024, [arXiv:2409.08949](https://arxiv.org/abs/2409.08949)):
Prometheus node telemetry joined to Slurm job metadata, one row per
(node, timestamp) per job.

## The notebooks

Run them in order; each one caches what the next one needs.

| notebook | what it does | runtime |
|---|---|---|
| `01_population.ipynb` | Census of the whole table, the population funnel, jobs and snapshots per label, job durations, telemetry cadence and gaps, node sharing, temporal drift, hardware and queue context, column health. Writes `data/jobs.parquet`. | ~3 min |
| `02_windows.ipynb` | Builds every 20-minute window and its label, then looks at what the aggregated features contain: box plots per metric and per label, single-feature separability, redundancy, precursor curves, horizon sensitivity. Writes `data/windows.parquet`. | ~6 min |
| `03_use_case.ipynb` | Turns those measurements into ceilings: observability, lead time, node-hours burned and recoverable, a telemetry-free baseline, and the per-state verdict. | <1 min |

Every notebook starts with the same ~35-line setup cell, repeated verbatim, so
each can be read and run on its own. The notebooks share **data**, never code —
there are no `.py` modules to chase.

## The pipeline constants

```python
WINDOW     = 40     # snapshots per window -> 20 minutes at a 30 s cadence
HORIZON_H  = 2.0    # a snapshot is positive within 2 h of a failing job's end
GAP_TOL_S  = 60.0   # reject any window containing a larger gap
STRIDE     = 10     # keep every 10th window start inside a series
TRAIN_FRAC = 0.8    # temporal split: the first 80 % of jobs by end_date
```

A window is 40 consecutive snapshots from one `(job, node)` series, summarised by
min/max/mean/std of 14 metrics — 56 features. A window is positive when at least
half its snapshots fall within 2 h of the end of a job that failed. The dataset
is never collapsed to one row per job: the unit of prediction is a 20-minute
window during a running job.

## What it found

- **The population.** 93,950,875 snapshots and 293,891 jobs in the full table;
  27,287 are GPU jobs, 23,618 after dropping CANCELLED. 51 nodes, 68 days
  (2022-08-25 → 2022-11-01). 26.7 % of jobs end in a failure state.

- **Coverage is a hard constraint.** Two thirds of failing jobs are over before a
  single 20-minute window can be formed. Median FAILED runtime is **1 minute**.
  Only 29.7 % of failures are observable with 20 minutes of lead time.

- **Node telemetry is shared.** 78 % of snapshots belong to a node-timestamp that
  serves more than one job, and the telemetry attached to them is byte-identical
  (verified). **56 % of positive windows** therefore carry the same feature values
  as a job that is *not* failing, which caps recall near **44 %** for any model
  built on node-level features. The exclusive queues (`gpu`, `gpu_titanrtx`) are
  the exception — and 7.8 % of the jobs.

- **TIMEOUT is a walltime problem, not a node problem.** 92.6 % of TIMEOUT jobs
  run to within 60 s of a whole hour, against 1.9 % of COMPLETED jobs, and the
  most common runtimes are 1, 2, 3, 24, 48 and 120 hours — values a user types by
  hand. TIMEOUT is also 83 % of the wasted node-hours, so it dominates any pooled
  model.

- **The horizon caps the value, more than coverage does.** Failures burn 37,249
  node-hours (39 % of the cluster). A perfect detector alerting at the first
  *positive* window recovers 7.8 % of that; one allowed to alert at the job's
  first window would recover 97 %. The 2-hour horizon alone puts 89 % of the
  prize out of reach.

- **No single feature separates the classes.** The best of the 56 reaches
  |AUC − 0.5| = 0.12, the median 0.05, and 30 of 56 are near-duplicates of another
  feature (|ρ| > 0.95). `std` separates least of the four aggregations.

`data/facts_all.json` holds every headline number, written by the notebooks, so
nothing has to be transcribed by hand.

## Layout

```
eda/
  01_population.ipynb  02_windows.ipynb  03_use_case.ipynb
  data/       cached parquet + facts_*.json   (git-ignored, rebuilt by running)
  figures/    every figure as PNG             (committed)
```

The notebooks expect the dataset at `../dataset/prom_slurm_joined/` and
`../dataset/node_hardware_info.parquet`, and write only inside `eda/`.

## Requirements

Python 3.13 with `pandas`, `numpy`, `pyarrow`, `scipy`, `scikit-learn`,
`matplotlib`, and `jupyter`. No seaborn, no project-local modules.

Peak memory is about 2.5 GB in notebook 02; the others stay under 1 GB. The
whole GPU population is read at full fidelity — nothing is sampled — because
`gpu_node = 1` rows occupy only 29 of the 200 parquet parts, so an Arrow
predicate skips ~85 % of the file.

## Cross-checks

Notebook 01 recomputes several quantities that an earlier, independently written
implementation in this project also produced (`pipeline_v2/results/audit.json`).
They agree exactly: 11,144,842 snapshots, 23,618 jobs, 51 nodes, a 30 s cadence,
and a train/test split at 2022-10-19 11:50:16. Two apparent disagreements
reconcile once the definition is pinned down — 78.7 % of failing jobs are shorter
than the horizon on the telemetry-span basis versus 77.3 % on true Slurm runtime,
and 152 jobs are multi-node according to Slurm while 123 actually report
telemetry from more than one node.
