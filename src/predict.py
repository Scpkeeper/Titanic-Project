"""
Prediction Module
==================
Load trained model, generate predictions on test set,
save submission.csv in Kaggle format (PassengerId, Survived).

CLI usage:
    python src/predict.py --config configs/config.yaml --model outputs/model.pkl
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def prepare_test_data(
    data_dir: str,
    preprocessor,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess test data using a fitted preprocessor.

    Parameters
    ----------
    data_dir : str
        Path to data directory containing test.csv.
    preprocessor : TitanicPreprocessor
        Fitted preprocessor instance (fit on training data).

    Returns
    -------
    tuple of (X_test, passenger_ids)
    """
    from src.data_loader import load_data
    from src.feature_engineering import FeatureEngineer

    # Load test data
    test_df = load_data(data_dir, "test.csv", backend="pandas")
    logger.info(f"Test data shape: {test_df.shape}")

    # Feature engineering (same as training)
    engineer = FeatureEngineer(create_interactions=True)
    test_df = engineer.transform(test_df)

    # Preprocessing with fitted preprocessor
    test_processed = preprocessor.transform(test_df)

    # Get feature columns
    feature_names = preprocessor.get_feature_names(test_df)
    exclude = {"PassengerId", "Survived"}
    feature_cols = [c for c in feature_names if c not in exclude]

    X_test = test_processed[feature_cols].select_dtypes(include=[np.number])
    passenger_ids = test_processed["PassengerId"]

    logger.info(f"Test features shape: {X_test.shape}")
    return X_test, passenger_ids


def predict(
    model_path: str,
    config_path: str,
) -> pd.DataFrame:
    """
    Generate predictions on test set and return submission DataFrame.

    Parameters
    ----------
    model_path : str
        Path to trained model pickle file (.pkl).
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    pd.DataFrame
        Submission dataframe with columns: PassengerId, Survived.
    """
    config = load_config(config_path)

    data_dir = config.get("data_dir", "./data")
    output_dir = Path(config.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fitted preprocessor — we need it for test preprocessing
    # For simplicity, re-fit on train data (in production, save/load preprocessor too)
    from src.data_loader import load_data
    from src.preprocessor import TitanicPreprocessor
    from src.feature_engineering import FeatureEngineer

    train_df = load_data(data_dir, "train.csv", backend="pandas")
    engineer = FeatureEngineer(create_interactions=True)
    train_df = engineer.transform(train_df)
    preprocessor = TitanicPreprocessor()
    preprocessor.fit(train_df)

    # Prepare test data
    X_test, passenger_ids = prepare_test_data(data_dir, preprocessor)

    # Scale features (use scaler fit on training data)
    scaler = StandardScaler()
    train_processed = preprocessor.transform(train_df)
    train_features = train_processed.select_dtypes(include=[np.number])
    train_features = train_features.drop(
        columns=["PassengerId", "Survived"], errors="ignore"
    )
    X_train_scaled = scaler.fit_transform(
        train_features.loc[:, X_test.columns] if len(train_features.columns) > 0 else train_features
    )
    # Align test columns with training columns
    common_cols = list(set(X_train_scaled.shape[1]) if hasattr(X_train_scaled, 'shape') else [])
    X_test_aligned = X_test.reindex(columns=X_test.columns, fill_value=0)
    X_test_scaled = scaler.transform(X_test_aligned)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Generate predictions
    predictions = model.predict(X_test_scaled)
    predictions = predictions.astype(int)

    # Create submission DataFrame
    submission = pd.DataFrame({
        "PassengerId": passenger_ids.values.astype(int),
        "Survived": predictions,
    })

    # Sanity check
    survival_rate = submission["Survived"].mean()
    logger.info(f"Predicted survival rate: {survival_rate:.4f} (~38% expected)")

    # Save submission
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    logger.info(f"Saved submission to {output_path}")

    print("\n" + "=" * 50)
    print("SUBMISSION SUMMARY")
    print("=" * 50)
    print(f"Total passengers: {len(submission)}")
    print(f"Predicted survived: {submission['Survived'].sum()}")
    print(f"Predicted died: {len(submission) - submission['Survived'].sum()}")
    print(f"Survival rate: {survival_rate:.4f} ({survival_rate*100:.1f}%)")
    print(f"Output file: {output_path}")
    print("=" * 50 + "\n")

    # Preview first few rows
    print(submission.head(10).to_string(index=False))

    return submission


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Titanic ML Prediction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/model.pkl",
        help="Path to trained model (.pkl).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    predict(args.model, args.config)


if __name__ == "__main__":
    main()
