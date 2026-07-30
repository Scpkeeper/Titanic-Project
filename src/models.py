"""
Models Module
==============
Implement multiple classifiers for the Titanic competition.

Classifiers included:
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost / LightGBM)
- Support Vector Machine
- Simple Neural Network (MLP)

Each classifier includes hyperparameter grids for tuning.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

LOGISTIC_REGRESSION_PARAMS = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [200, 500],
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}

GRADIENT_BOOSTING_PARAMS_XGB = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

GRADIENT_BOOSTING_PARAMS_LGBM = {
    "n_estimators": [100, 200],
    "num_leaves": [31, 50],
    "learning_rate": [0.01, 0.05, 0.1],
    "feature_fraction": [0.8, 1.0],
}

SVM_PARAMS = {
    "C": [0.1, 1.0, 10.0],
    "kernel": ["rbf", "linear", "poly"],
    "gamma": ["scale", "auto"],
}

MLP_PARAMS = {
    "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001],
    "max_iter": [500, 1000],
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(model_name: str, random_state: int = 42) -> Any:
    """
    Instantiate a model by name.

    Parameters
    ----------
    model_name : str
        One of: 'logistic_regression', 'random_forest', 'xgboost',
        'lightgbm', 'svm', 'mlp'.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    estimator
        Scikit-learn compatible estimator.

    Raises
    ------
    ValueError
        If model_name is unknown.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    models = {
        "logistic_regression": LogisticRegression(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state),
        "mlp": MLPClassifier(random_state=random_state),
    }

    # Optional models
    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
    except ImportError:
        logger.warning("XGBoost not installed. Skipping xgboost model.")

    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        logger.warning("LightGBM not installed. Skipping lightgbm model.")

    if model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    logger.info(f"Created model: {model_name}")
    return models[model_name]


def get_param_grid(model_name: str) -> dict[str, list]:
    """
    Return hyperparameter grid for a given model.

    Parameters
    ----------
    model_name : str
        Model name (same as get_model keys).

    Returns
    -------
    dict
        Parameter grid suitable for GridSearchCV.
    """
    param_grids = {
        "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
        "random_forest": RANDOM_FOREST_PARAMS,
        "gradient_boosting": GRADIENT_BOOSTING_PARAMS_LGBM,
        "xgboost": GRADIENT_BOOSTING_PARAMS_XGB,
        "lightgbm": GRADIENT_BOOSTING_PARAMS_LGBM,
        "svm": SVM_PARAMS,
        "mlp": MLP_PARAMS,
    }

    if model_name not in param_grids:
        raise ValueError(f"No parameter grid defined for '{model_name}'")

    return param_grids[model_name]


def get_available_models() -> list[str]:
    """Return list of available model names."""
    names = [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "svm",
        "mlp",
    ]
    try:
        import xgboost
        names.append("xgboost")
    except ImportError:
        pass
    try:
        import lightgbm
        names.append("lightgbm")
    except ImportError:
        pass
    return names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Available models:")
    for name in get_available_models():
        model = get_model(name)
        grid = get_param_grid(name)
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"  {name}: {type(model).__name__}, {n_combos} param combos")
