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
| `01_population.ipynb` | *Part 1* — the dataset as it is: census, population funnel, states, durations, cadence and gaps, hardware and queues, column health. *Part 2* — four checks prompted by a downstream result: observability, co-tenancy, TIMEOUT, drift. Writes `data/jobs.parquet`. | ~3 min |
| `02_windows.ipynb` | *Part 1* — builds every 20-minute window and its label. *Part 2* — four questions in a chain: is the label determined by the features, does any single feature separate the classes, do the features move as the end approaches, would another horizon change the answer. Writes `data/windows.parquet`. | ~6 min |
| `03_use_case.ipynb` | One oracle's ceilings: which failures are observable and how early, a node-hour cascade from the prize down to what is recoverable, a telemetry-free floor, and the per-state verdict. | <1 min |

Each notebook opens with the same short setup cell, so it can be read and run on
its own. The notebooks share **data**, never code — there are no `.py` modules to
chase.

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
  node-hours (39 % of the cluster). Coverage costs under 1 % of that, because the
  failures nobody can see are the short ones. An oracle allowed to alert at a
  job's first window recovers 97 %; restricted to alerting inside the 2-hour
  horizon it recovers 7.8 %. The horizon alone puts 89 % of the prize out of
  reach.

- **No single feature separates the classes.** The best of the 56 reaches
  |AUC − 0.5| = 0.12, the median 0.05, and 30 of 56 are near-duplicates of another
  feature (|ρ| > 0.95). `std` separates least of the four aggregations.

- **What separation exists is a level, not a trajectory.** Failing and completing
  jobs are offset from each other six hours before the end, and the offset barely
  moves as the end approaches — the features report what kind of job this is, not
  that something is about to go wrong with it. `NODE_FAIL` (a collapse in the last
  10–15 minutes) and `OUT_OF_MEMORY` are the exceptions, and they are the two
  smallest states.

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
