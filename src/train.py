"""
Training Module
================
Cross-validation, hyperparameter tuning, and model training pipeline.

Supports:
- Stratified K-Fold CV (n_splits=5)
- GridSearchCV / Optuna hyperparameter tuning
- Feature scaling (StandardScaler)
- Comprehensive metrics: accuracy, precision, recall, F1, ROC-AUC
- CLI usage: python src/train.py --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def prepare_data(
    data_dir: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and split training data.

    Parameters
    ----------
    data_dir : str
        Path to data directory containing train.csv.
    random_state : int
        Random seed.
    test_size : float
        Fraction of data for validation.

    Returns
    -------
    tuple of (X_train, X_val, y_train, y_val)
    """
    from src.data_loader import load_data
    from src.preprocessor import TitanicPreprocessor
    from src.feature_engineering import FeatureEngineer

    # Load raw data
    df = load_data(data_dir, "train.csv", backend="pandas")
    logger.info(f"Raw data shape: {df.shape}")

    # Feature engineering
    engineer = FeatureEngineer(create_interactions=True)
    df = engineer.transform(df)

    # Preprocessing
    preprocessor = TitanicPreprocessor()
    df_processed = preprocessor.fit_transform(df)

    # Get feature columns
    feature_names = preprocessor.get_feature_names(df)
    exclude = {"PassengerId", "Survived"}
    feature_cols = [c for c in feature_names if c not in exclude]

    X = df_processed[feature_cols].select_dtypes(include=[np.number])
    y = df_processed["Survived"]

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a trained model on test data.

    Returns dict with accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = model.predict(X_test)
    y_proba = None

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> tuple[object, dict[str, float]]:
    """
    Train a single model with hyperparameter tuning.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g., 'random_forest').
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    config : dict
        Configuration dictionary.

    Returns
    -------
    tuple of (trained_model, best_metrics)
    """
    from src.models import get_model, get_param_grid

    random_state = config.get("random_state", 42)
    cv_folds = config.get("cv_folds", 5)

    # Get model and param grid
    model = get_model(model_name, random_state=random_state)
    param_grid = get_param_grid(model_name)

    # Cross-validation with GridSearchCV
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    logger.info(f"Training {model_name} with GridSearchCV ({cv_folds}-fold CV)...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    logger.info(
        f"{model_name}: Best CV Accuracy = {best_cv_score:.4f}, "
        f"Best Params = {best_params}"
    )

    # CV metrics
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="accuracy")

    metrics = {
        "model_name": model_name,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "best_params": str(best_params),
    }

    return best_model, metrics


def run_training_pipeline(config_path: str) -> dict:
    """
    Run the complete training pipeline.

    Trains all available models, compares results, saves the best model.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    dict
        Summary of all models' performance.
    """
    config = load_config(config_path)

    data_dir = config.get("data_dir", "./data")
    random_state = config.get("random_state", 42)
    test_size = config.get("test_size", 0.2)
    output_dir = Path(config.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(
        data_dir, random_state=random_state, test_size=test_size
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train all models
    from src.models import get_available_models

    all_results = []
    best_model = None
    best_score = 0.0
    best_model_name = ""

    for model_name in get_available_models():
        try:
            model, cv_metrics = train_model(
                model_name, X_train_scaled, y_train, config
            )
            val_metrics = evaluate_model(model, X_val_scaled, y_val)

            result = {**cv_metrics, **val_metrics}
            all_results.append(result)

            logger.info(
                f"{model_name}: CV Acc={result['cv_accuracy_mean']:.4f}, "
                f"Val Acc={result['accuracy']:.4f}, "
                f"F1={result['f1']:.4f}"
            )

            if result["accuracy"] > best_score:
                best_score = result["accuracy"]
                best_model = model
                best_model_name = model_name

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue

    # Save best model
    import joblib
    model_path = output_dir / "model.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model ({best_model_name}) to {model_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'CV Acc':>10} {'Val Acc':>10} {'F1':>8} {'ROC-AUC':>8}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x.get("accuracy", 0), reverse=True):
        name = r.get("model_name", "?")[:24]
        cv_acc = r.get("cv_accuracy_mean", 0)
        val_acc = r.get("accuracy", 0)
        f1 = r.get("f1", 0)
        auc = r.get("roc_auc", 0)
        print(f"{name:<25} {cv_acc:>10.4f} {val_acc:>10.4f} {f1:>8.4f} {auc:>8.4f}")
    print("-" * 70)
    print(f"Best model: {best_model_name} (Val Acc = {best_score:.4f})")
    print("=" * 70 + "\n")

    return {"results": all_results, "best_model": best_model_name}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Titanic ML Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_training_pipeline(args.config)


if __name__ == "__main__":
    main()
