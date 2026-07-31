from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from titanic_b_pipeline import (
    RAW_FEATURE_COLUMNS,
    TitanicFeatureEngineer,
    make_model_pipeline,
    make_transform_pipeline,
    validate_raw_data,
)


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "source" / "train.csv"
TEST_PATH = ROOT / "source" / "test.csv"


class TitanicBPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = pd.read_csv(TRAIN_PATH)
        cls.test = pd.read_csv(TEST_PATH)
        cls.X = cls.train[RAW_FEATURE_COLUMNS]
        cls.y = cls.train["Survived"]

    def test_01_raw_data_contract(self) -> None:
        validate_raw_data(self.train, self.test)
        self.assertEqual(self.train.shape, (891, 12))
        self.assertEqual(self.test.shape, (418, 11))

    def test_02_feature_engineering_removes_missing(self) -> None:
        engineer = TitanicFeatureEngineer().fit(self.X)
        train_clean = engineer.transform(self.X)
        test_clean = engineer.transform(self.test)
        self.assertEqual(int(train_clean.isna().sum().sum()), 0)
        self.assertEqual(int(test_clean.isna().sum().sum()), 0)

    def test_03_row_order_and_ids_are_preserved(self) -> None:
        engineer = TitanicFeatureEngineer().fit(self.X)
        clean = engineer.transform(self.X)
        self.assertListEqual(
            clean["PassengerId"].tolist(),
            self.train["PassengerId"].tolist(),
        )

    def test_04_expected_features_are_created(self) -> None:
        clean = TitanicFeatureEngineer().fit_transform(self.X)
        expected = {
            "Title",
            "FamilySize",
            "IsAlone",
            "TicketPrefix",
            "EmbarkedFilled",
            "CabinDeck",
            "AgeFilled",
            "FareFilled",
        }
        self.assertTrue(expected.issubset(clean.columns))

    def test_05_fit_does_not_use_target(self) -> None:
        engineer_a = TitanicFeatureEngineer().fit(self.X, self.y)
        engineer_b = TitanicFeatureEngineer().fit(self.X, 1 - self.y)
        self.assertEqual(
            engineer_a.age_group_medians_,
            engineer_b.age_group_medians_,
        )
        self.assertEqual(
            engineer_a.frequent_ticket_prefixes_,
            engineer_b.frequent_ticket_prefixes_,
        )

    def test_06_unknown_categories_are_safe(self) -> None:
        pipeline = make_transform_pipeline(k=12)
        pipeline.fit(self.X, self.y)
        modified = self.test.head(3).copy()
        modified.loc[modified.index[0], "Sex"] = "unknown"
        modified.loc[modified.index[1], "Embarked"] = "X"
        transformed = pipeline.transform(modified)
        self.assertEqual(transformed.shape, (3, 12))
        self.assertTrue(np.isfinite(transformed).all())

    def test_07_cross_validation_runs_from_raw_data(self) -> None:
        pipeline = make_model_pipeline(k=12)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline,
            self.X,
            self.y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(scores.mean()), 0.75)

    def test_08_exported_clean_data_contract(self) -> None:
        train_clean = pd.read_csv(ROOT / "data" / "Titanic_train_clean.csv")
        test_clean = pd.read_csv(ROOT / "data" / "Titanic_test_clean.csv")
        self.assertEqual(train_clean.shape[0], 891)
        self.assertEqual(test_clean.shape[0], 418)
        self.assertEqual(int(train_clean.isna().sum().sum()), 0)
        self.assertEqual(int(test_clean.isna().sum().sum()), 0)
        self.assertEqual(
            train_clean.columns.drop("Survived").tolist(),
            test_clean.columns.tolist(),
        )

    def test_09_exported_model_ready_alignment(self) -> None:
        train_ready = pd.read_csv(ROOT / "data" / "Titanic_train_model_ready.csv")
        test_ready = pd.read_csv(ROOT / "data" / "Titanic_test_model_ready.csv")
        train_features = train_ready.columns.drop(["PassengerId", "Survived"]).tolist()
        test_features = test_ready.columns.drop("PassengerId").tolist()
        self.assertListEqual(train_features, test_features)
        self.assertEqual(int(train_ready.isna().sum().sum()), 0)
        self.assertEqual(int(test_ready.isna().sum().sum()), 0)
        self.assertTrue(
            all(
                pd.api.types.is_numeric_dtype(train_ready[column])
                for column in train_ready
            )
        )

    def test_10_quality_report_passes(self) -> None:
        report = json.loads(
            (ROOT / "reports" / "quality_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["train_test_model_features_aligned"])
        self.assertEqual(report["clean_missing_values"], {"train": 0, "test": 0})
        self.assertGreater(report["nested_outer_accuracy_mean"], 0.78)


if __name__ == "__main__":
    unittest.main(verbosity=2)
