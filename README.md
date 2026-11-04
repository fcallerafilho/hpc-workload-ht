# Does Early Telemetry Add to the Job Request?

Code for the undergraduate thesis *Does Early Telemetry Add to the Job Request? Ranking Metric
Groups for GPU Job Failures on Marconi100* (Electronic Engineering, UNIFEI).

We test whether telemetry from the first 20 minutes of a GPU job on Marconi100 predicts failure
better than the job request alone (B0). It does not: none of the ten metric groups passes the
relevance rule, for failures or for timeouts.

## Notebooks

Run in order.

| Notebook | Paper section |
|---|---|
| [`01_cohort`](notebooks/01_cohort.ipynb) | II-B, III-A, III-B |
| [`02_channels`](notebooks/02_channels.ipynb) | III-C, Table I |
| [`03_clock`](notebooks/03_clock.ipynb) | III-D |
| [`04_completeness`](notebooks/04_completeness.ipynb) | III-E, Tables II–III |
| [`05_features`](notebooks/05_features.ipynb) | IV-A to IV-C |
| [`06_splits`](notebooks/06_splits.ipynb) | IV-D, IV-F, IV-G |
| [`07_models`](notebooks/07_models.ipynb) | IV-E |
| [`08_results`](notebooks/08_results.ipynb) | IV-F, IV-G, V, Tables IV–V, Fig. 1 |

Methodological decisions were recorded, with dates, before the results they affect:
[`DECISOES.md`](DECISOES.md) (in Portuguese).

## Running

Data (not included):

* PM100 job table, [doi:10.5281/zenodo.8129258](https://doi.org/10.5281/zenodo.8129258), at
  `dataset/pm100/job_table.parquet`
* M100 ExaData, May–Oct 2020, [doi:10.5281/zenodo.7588815](https://doi.org/10.5281/zenodo.7588815),
  extracted to `dataset/m100/year_month=20-05` … `20-10`

Set `TCC_DATASET` to use another location.

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

Notebook 02 reads the raw telemetry (40–60 min) and notebook 07 fits the models (~20 min); both
cache their output in `data/`.

## Layout

```
notebooks/   analysis, one notebook per part of the paper
tcc/         shared code: paths and constants, temporal splits
results/     paper tables as CSV
```
