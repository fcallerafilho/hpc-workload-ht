"""Pass 2 parameter grids (only defined for the grid-able models in the spec).

get_grids(scale_pos_weight) returns {model_name: GridSpec}. A model selected for
Pass 2 that has no entry here (e.g. CatBoost, ExtraTrees) is skipped with a note.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from . import config


@dataclass
class GridSpec:
    make_estimator: Callable[[], Any]
    param_grid: dict
    needs_scaling: bool = False


def get_grids(scale_pos_weight: float) -> dict[str, GridSpec]:
    seed = config.SEED
    grids: dict[str, GridSpec] = {}

    from lightgbm import LGBMClassifier
    grids["LightGBM"] = GridSpec(
        lambda: LGBMClassifier(n_jobs=-1, random_state=seed, verbose=-1),
        {
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
            "min_child_samples": [20, 100, 500],
            "scale_pos_weight": [scale_pos_weight],
        },
    )

    from xgboost import XGBClassifier
    grids["XGBoost"] = GridSpec(
        lambda: XGBClassifier(n_jobs=-1, random_state=seed, eval_metric="aucpr"),
        {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 500],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [scale_pos_weight],
            "tree_method": ["hist"],
        },
    )

    # CatBoost grid is NOT in the written spec (which predates CatBoost topping
    # Pass 1). Added here to tune the #1 model: the three highest-impact knobs
    # (tree depth, learning rate, boosting rounds), class weights fixed to
    # Balanced. 12 candidates x 3 folds = 36 fits.
    from catboost import CatBoostClassifier
    grids["CatBoost"] = GridSpec(
        lambda: CatBoostClassifier(auto_class_weights="Balanced", verbose=0,
                                   random_seed=seed, thread_count=-1,
                                   allow_writing_files=False),
        {
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "iterations": [300, 500],
        },
    )

    grids["RandomForestClassifier"] = GridSpec(
        lambda: RandomForestClassifier(n_jobs=-1, random_state=seed),
        {
            "n_estimators": [200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 20, 100],
            "max_features": ["sqrt", 0.5],
            "class_weight": ["balanced", "balanced_subsample"],
        },
    )

    grids["HistGradientBoostingClassifier"] = GridSpec(
        lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
        {
            "max_leaf_nodes": [15, 31, 63],
            "learning_rate": [0.05, 0.1],
            "max_iter": [200, 500],
            "l2_regularization": [0.0, 1.0],
            "min_samples_leaf": [20, 100],
        },
    )

    from imblearn.ensemble import BalancedRandomForestClassifier
    grids["BalancedRandomForestClassifier"] = GridSpec(
        lambda: BalancedRandomForestClassifier(replacement=True, bootstrap=False,
                                               n_jobs=-1, random_state=seed),
        {
            "n_estimators": [200, 500],
            "max_depth": [None, 10, 20],
            "sampling_strategy": [0.1, 0.5, 1.0],
            "min_samples_leaf": [1, 20, 100],
        },
    )

    grids["MLPClassifier"] = GridSpec(
        lambda: MLPClassifier(early_stopping=True, max_iter=200, random_state=seed),
        {
            "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 1e-4],
        },
        needs_scaling=True,
    )

    return grids
