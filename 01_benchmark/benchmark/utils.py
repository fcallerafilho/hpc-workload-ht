"""Shared utilities: seeding, timing, peak-RAM sampling, logging, CSV append."""
from __future__ import annotations

import csv
import logging
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psutil

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Fix every RNG we can reach. sklearn/lgbm/xgb/catboost also take random_state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "benchmark") -> logging.Logger:
    """Logger writing to both the console and results/benchmark.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    box = {"t": None}
    try:
        yield lambda: (box["t"] if box["t"] is not None else time.perf_counter() - start)
    finally:
        box["t"] = time.perf_counter() - start


class PeakMemorySampler:
    """Polls this process's RSS in a background thread; records the peak in MB.

    Used to detect which models approach the 16 GB laptop RAM limit.
    """

    def __init__(self, interval: float = 0.25):
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
            except Exception:
                rss = 0
            if rss > self._peak:
                self._peak = rss
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakMemorySampler":
        self._peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 ** 2)


def append_row(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row to a CSV, writing the header if the file is new.

    Flushed immediately so a crash mid-benchmark preserves completed rows.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
