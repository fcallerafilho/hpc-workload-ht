"""Model benchmark for proactive HPC node-failure prediction (TCC UNIFEI).

Runnable as modules:
    python -m benchmark.data_pipeline      # build .npy artifacts (once)
    python -m benchmark.run_pass1          # 15 base models -> results.csv
    python -m benchmark.run_pass2          # GridSearchCV on top-K from Pass 1
"""
