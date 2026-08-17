from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.family_ticket_model import FamilyTicketCatBoostClassifier
from titanic_b_pipeline import RAW_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


class FamilyTicketModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        train = pd.read_csv(ROOT / "source" / "train.csv")
        cls.X = train.loc[:179, RAW_FEATURE_COLUMNS].copy()
        cls.y = train.loc[:179, "Survived"].astype(int).to_numpy()
        cls.holdout = train.loc[180:199, RAW_FEATURE_COLUMNS].copy()

    def make_model(self) -> FamilyTicketCatBoostClassifier:
        return FamilyTicketCatBoostClassifier(
            iterations=30,
            depth=4,
            learning_rate=0.08,
            signal_folds=3,
            smoothing=2.0,
            min_support=2,
            random_state=42,
        )

    def test_estimator_accepts_raw_rows_and_returns_valid_probabilities(self) -> None:
        model = self.make_model().fit(self.X, self.y)
        probability = model.predict_proba(self.holdout)
        prediction = model.predict(self.holdout)

        self.assertEqual(probability.shape, (20, 2))
        self.assertTrue(np.isfinite(probability).all())
        np.testing.assert_allclose(probability.sum(axis=1), np.ones(20))
        self.assertTrue(set(np.unique(prediction)).issubset({0, 1}))
        self.assertNotIn("PassengerId", model.model_feature_names_)

    def test_fixed_seed_is_deterministic(self) -> None:
        first = self.make_model().fit(self.X, self.y).predict_proba(self.holdout)
        second = self.make_model().fit(self.X, self.y).predict_proba(self.holdout)
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_serialized_model_reproduces_predictions(self) -> None:
        model = self.make_model().fit(self.X, self.y)
        expected = model.predict_proba(self.holdout)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Setting the shape on a NumPy array has been deprecated",
                    category=DeprecationWarning,
                )
                loaded = joblib.load(path)
            actual = loaded.predict_proba(self.holdout)
        np.testing.assert_allclose(expected, actual, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
