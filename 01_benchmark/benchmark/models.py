"""Pass 1 base model configurations.

get_base_models(scale_pos_weight) returns an ordered list of ModelSpec, one per
row of the benchmark's Pass-1 tables. `needs_scaling` marks the estimators that
consume the StandardScaler-transformed X (linear models, NB, MLP); tree/boosting
models ignore scaling and read the raw float32 matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from . import config


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator: Any
    needs_scaling: bool = False
    params: dict | None = None  # compact record of the base config for results.csv


def get_base_models(scale_pos_weight: float) -> list[ModelSpec]:
    seed = config.SEED
    specs: list[ModelSpec] = []

    # -- Tier 0: baselines -------------------------------------------------- #
    specs.append(ModelSpec(
        "DummyClassifier", "baseline",
        DummyClassifier(strategy="prior", random_state=seed),
        params={"strategy": "prior"},
    ))
    specs.append(ModelSpec(
        "GaussianNB", "baseline", GaussianNB(), needs_scaling=True, params={},
    ))
    specs.append(ModelSpec(
        "LogisticRegression", "linear",
        LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs",
                           random_state=seed),
        needs_scaling=True,
        params={"class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs"},
    ))

    # -- Tier 1: trees and ensembles --------------------------------------- #
    specs.append(ModelSpec(
        "DecisionTreeClassifier", "tree",
        DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=seed),
        params={"class_weight": "balanced", "max_depth": 10},
    ))
    specs.append(ModelSpec(
        "RandomForestClassifier", "forest",
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=seed),
        params={"n_estimators": 300, "class_weight": "balanced"},
    ))
    specs.append(ModelSpec(
        "ExtraTreesClassifier", "forest",
        ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                             n_jobs=-1, random_state=seed),
        params={"n_estimators": 300, "class_weight": "balanced"},
    ))
    specs.append(ModelSpec(
        "HistGradientBoostingClassifier", "boosting",
        HistGradientBoostingClassifier(class_weight="balanced", max_iter=300,
                                       random_state=seed),
        params={"class_weight": "balanced", "max_iter": 300},
    ))

    # LightGBM
    from lightgbm import LGBMClassifier
    specs.append(ModelSpec(
        "LightGBM", "boosting",
        LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                       scale_pos_weight=scale_pos_weight, n_jobs=-1,
                       random_state=seed, verbose=-1),
        params={"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.1,
                "scale_pos_weight": round(scale_pos_weight, 4)},
    ))

    # XGBoost
    from xgboost import XGBClassifier
    specs.append(ModelSpec(
        "XGBoost", "boosting",
        XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      scale_pos_weight=scale_pos_weight, tree_method="hist",
                      n_jobs=-1, random_state=seed, eval_metric="aucpr"),
        params={"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
                "scale_pos_weight": round(scale_pos_weight, 4), "tree_method": "hist"},
    ))

    # CatBoost
    from catboost import CatBoostClassifier
    specs.append(ModelSpec(
        "CatBoost", "boosting",
        CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                           auto_class_weights="Balanced", verbose=0,
                           random_seed=seed, thread_count=-1, allow_writing_files=False),
        params={"iterations": 300, "depth": 6, "learning_rate": 0.1,
                "auto_class_weights": "Balanced"},
    ))

    # -- Tier 2: imbalance-aware ------------------------------------------- #
    from imblearn.ensemble import (
        BalancedRandomForestClassifier,
        EasyEnsembleClassifier,
        RUSBoostClassifier,
    )
    specs.append(ModelSpec(
        "BalancedRandomForestClassifier", "imbalance",
        BalancedRandomForestClassifier(n_estimators=300, sampling_strategy="auto",
                                       replacement=True, bootstrap=False,
                                       n_jobs=-1, random_state=seed),
        params={"n_estimators": 300, "sampling_strategy": "auto",
                "replacement": True, "bootstrap": False},
    ))
    specs.append(ModelSpec(
        "EasyEnsembleClassifier", "imbalance",
        # n_jobs=1 (not -1): this is the only bagging ensemble that parallelizes
        # across *processes* (loky), which duplicates arrays per worker and
        # exhausts the 16 GB laptop RAM. Single-process is RAM-safe; n_jobs does
        # not affect the fitted model (per-estimator seeds derive from random_state).
        EasyEnsembleClassifier(n_estimators=20, n_jobs=1, random_state=seed),
        params={"n_estimators": 20},
    ))
    specs.append(ModelSpec(
        "RUSBoostClassifier", "imbalance",
        RUSBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=seed),
        params={"n_estimators": 200, "learning_rate": 0.1},
    ))

    # -- Tier 3: non-tree --------------------------------------------------- #
    specs.append(ModelSpec(
        "MLPClassifier", "neural",
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, early_stopping=True,
                      max_iter=200, random_state=seed),
        needs_scaling=True,
        params={"hidden_layer_sizes": [64, 32], "alpha": 1e-4,
                "early_stopping": True, "max_iter": 200},
    ))
    specs.append(ModelSpec(
        "SGDClassifier", "linear",
        SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000,
                      n_jobs=-1, random_state=seed),
        needs_scaling=True,
        params={"loss": "log_loss", "class_weight": "balanced", "max_iter": 1000},
    ))

    return specs
