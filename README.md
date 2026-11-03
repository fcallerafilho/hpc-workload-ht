# Does Early Telemetry Add to the Job Request?

Code for the undergraduate thesis *Does Early Telemetry Add to the Job Request? Ranking Metric
Groups for GPU Job Failures on Marconi100* (Electronic Engineering, Universidade Federal de
Itajubá).

**Question.** Which telemetry metrics, observed in the **first 20 minutes** of a GPU job on the
Marconi100 supercomputer, are associated with the job's failure once duration, requested resources
and failure type are controlled?

**Answer.** None. Ten metric groups were each compared with a baseline that holds only what the
user requested at submission (B0). Under a relevance rule fixed before any model was fitted, no
group is relevant for either target: (a) application and hardware failures, or (b) timeouts. The
models do use the telemetry, but what it carries is already in the job request.

## Every claim of the paper has a notebook

The notebooks in `notebooks/` are meant to be read in order. Each covers one part of the paper,
and every number the paper prints is checked with a `claim(...)` line that shows the computed
value next to the paper's text. The checks of each notebook are saved in `results/claims/`.

| Notebook | Paper | What it shows |
|---|---|---|
| [`01_cohort`](notebooks/01_cohort.ipynb) | II-B, III-A, III-B | PM100 job table, the 69,096-job cohort, the GPU filter (66,017 jobs), the two targets |
| [`02_channels`](notebooks/02_channels.ipynb) | III-C, Table I | One pass over the raw ExaData files: 19 channels, cadences, no counters, 2 constant channels; the 10 metric groups |
| [`03_clock`](notebooks/03_clock.ipynb) | III-D | Telemetry and job times share one clock; PM100's own power traces are offset by +2 h |
| [`04_completeness`](notebooks/04_completeness.ipynb) | III-E, Tables II–III | Fraction of expected readings in the window; the pre-fixed procedure that picks "≥ 1 reading" (24,107 jobs); selection by month and state |
| [`05_features`](notebooks/05_features.ipynb) | IV-A, IV-B, IV-C | Window series, the leakage check V1, 10 groups × 4 statistics, B0, placebos |
| [`06_splits`](notebooks/06_splits.ipynb) | IV-D, IV-F, IV-G | Temporal splits, floors of 100 positives, user concentration, the 21.6 % chance of a "stable" rank |
| [`07_models`](notebooks/07_models.ipynb) | IV-E | HGB, RF and LightGBM fitted on B0, B0 + each group and B0 + each placebo |
| [`08_results`](notebooks/08_results.ipynb) | IV-F, IV-G, V, Tables IV–V, Fig. 1 | Gains with user-level bootstrap intervals, rule A–E, Kendall's W, placebos, model agreement, selection, sensitivity, readings |

Claims that need no code are not in the notebooks: the related work (Section I), the description
of the machine and of the acquisition chain (Section III-A, from Antici et al. and Borghesi et
al.), and the reasoning of the Discussion.

`results/` holds the paper's tables as CSV files (`metric_groups.csv` for Table I,
`completeness_rules.csv` and `populations.csv` for Tables II–III, `performance.csv` for Table IV,
`final_table.csv` for Table V) plus `gains.csv` and `channel_inventory.csv`. The notebooks show each
table and Fig. 1 inline.

## Decisions

Every methodological choice (cohort limits, floors, the completeness procedure, B0, models,
placebos, splits, bootstrap, readings) was written down with a date **before** the results it
affects, in [`DECISOES.md`](DECISOES.md) (Portuguese; entries P1–P13). The single change made after
seeing results (P13: LightGBM joins RF in condition E) can only make the rule stricter, and both
versions are reported.

## Running it

**Data** (public, not in the repository), under `dataset/` or the folder named by the environment
variable `TCC_DATASET`:

* `dataset/pm100/job_table.parquet`: PM100, Antici et al., [Zenodo 10.5281/zenodo.8129258](https://doi.org/10.5281/zenodo.8129258);
* `dataset/m100/year_month=20-05` … `year_month=20-10`: M100 ExaData, Borghesi et al., the
  2020-03 to 2020-12 release, [Zenodo 10.5281/zenodo.7588815](https://doi.org/10.5281/zenodo.7588815),
  with each month's tar extracted in place (`plugin=ganglia_pub`, `plugin=ipmi_pub`,
  `plugin=job_table`).

**Environment:** Python 3.13 and `pip install -r requirements.txt`.

**Order:** run the notebooks 01 → 08 (`jupyter nbconvert --to notebook --execute --inplace
notebooks/0*.ipynb`, or one by one). Two steps are slow and cache their results in `data/`:
notebook 02 reads the raw telemetry (about 40–60 minutes, about 1 GB of memory) and
notebook 07 fits the models (about 20 minutes). Everything else takes seconds to a few minutes.
Seeds are fixed, so a re-run gives the same numbers.

## Layout

```
notebooks/   the eight notebooks above (committed with their outputs)
tcc/         what more than one notebook needs: config.py (paths, constants, the metric groups),
             splits.py (the temporal splits), claims.py (the check against the paper)
results/     the paper's tables as CSV, and the claim checks
data/        derived data written by the notebooks (not versioned)
dataset/     raw data (not versioned)
DECISOES.md  dated decision log
```
