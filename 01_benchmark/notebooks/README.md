# notebooks/ — traceable, step-by-step view

A parallel view of the benchmark implemented in `../benchmark/`, one notebook per stage.
**Hybrid design:** shared logic (constants, model list, grids, `evaluate()`) is imported
from `benchmark/`; the pipeline transformation, orchestration, and plots are inlined so the
methodology is visible. **Non-destructive:** reads `../artifacts/` and `../results/`, never
overwrites them.

| Notebook | What it traces |
|---|---|
| `00_setup.ipynb` | paths, seed, the 8 features, artifact sanity check |
| `01_data_pipeline.ipynb` | load → label → split → window, with intermediate outputs, a sample-job plot, and a validation against the saved artifacts |
| `02_pass1_baselines.ipynb` | the 15 base configs → PR-AUC ranking + bar chart |
| `03_pass2_gridsearch.ipynb` | the grids → base-vs-tuned → the CV-leakage finding |
| `04_analysis_findings.ipynb` | per-state recall, diagnostics, conclusions |

Run order is the numeric order. **Heavy work is gated** behind flags (default `False`):
`RUN_TRAINING` (02, ~2 h), `RUN_GRID` (03, ~4 h), `REBUILD_FULL` (01, ~2 min). With the
flags off, every notebook opens in seconds by loading precomputed `results/results.csv`
and `artifacts/`.
