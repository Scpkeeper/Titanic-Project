"""CatBoost estimator with cross-fitted Titanic family/ticket evidence."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.family_ticket_signal import (
    GROUP_SIGNAL_COLUMNS,
    GroupSignalMap,
    cross_fit_group_signals,
    normalize_group_keys,
)
from titanic_b_pipeline import RAW_FEATURE_COLUMNS, TitanicFeatureEngineer


CATEGORICAL_MODEL_COLUMNS = [
    "Pclass",
    "Sex",
    "AgeBand",
    "FamilyGroup",
    "FareBand",
    "EmbarkedFilled",
    "Title",
    "CabinDeck",
    "TicketPrefix",
    "SexPclass",
    "FamilyKey",
    "TicketKey",
]


class FamilyTicketCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Fit CatBoost without exposing a row's own label to group features."""

    def __init__(
        self,
        *,
        iterations: int = 500,
        depth: int = 5,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 5.0,
        random_strength: float = 0.5,
        signal_folds: int = 5,
        smoothing: float = 2.0,
        min_support: int = 2,
        decision_threshold: float = 0.5,
        use_group_signals: bool = True,
        random_state: int = 42,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.signal_folds = signal_folds
        self.smoothing = smoothing
        self.min_support = min_support
        self.decision_threshold = decision_threshold
        self.use_group_signals = use_group_signals
        self.random_state = random_state

    @staticmethod
    def _validate_raw(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FamilyTicketCatBoostClassifier requires a DataFrame")
        missing = sorted(set(RAW_FEATURE_COLUMNS).difference(X.columns))
        if missing:
            raise ValueError(f"Missing raw Titanic columns: {missing}")

    def _base_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        clean = self.feature_engineer_.transform(X[RAW_FEATURE_COLUMNS])
        matrix = clean.drop(columns=["PassengerId"]).copy()
        keys = normalize_group_keys(X)
        matrix["FamilyKey"] = keys["FamilyKey"].to_numpy()
        matrix["TicketKey"] = keys["TicketKey"].to_numpy()
        return matrix

    def _prepare_matrix(
        self, X: pd.DataFrame, signals: pd.DataFrame | None
    ) -> pd.DataFrame:
        matrix = self._base_matrix(X)
        if self.use_group_signals:
            if signals is None:
                raise ValueError("Group signals are required for this estimator")
            aligned = signals.copy()
            aligned.index = matrix.index
            matrix = pd.concat([matrix, aligned[GROUP_SIGNAL_COLUMNS]], axis=1)
        for column in CATEGORICAL_MODEL_COLUMNS:
            matrix[column] = matrix[column].astype(str)
        if matrix.isna().any().any():
            raise AssertionError("CatBoost model matrix contains missing values")
        return matrix

    def fit(
        self, X: pd.DataFrame, y: Sequence[int]
    ) -> "FamilyTicketCatBoostClassifier":
        self._validate_raw(X)
        labels = np.asarray(y, dtype=int)
        if len(labels) != len(X):
            raise ValueError("X and y must have the same number of rows")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("Survival labels must be binary")
        if not 0 < self.decision_threshold < 1:
            raise ValueError("decision_threshold must be between zero and one")

        raw = X[RAW_FEATURE_COLUMNS].copy()
        self.feature_engineer_ = TitanicFeatureEngineer().fit(raw)
        self.group_signal_map_ = GroupSignalMap(
            smoothing=self.smoothing, min_support=self.min_support
        ).fit(raw, labels)
        train_signals = None
        if self.use_group_signals:
            train_signals = cross_fit_group_signals(
                raw,
                labels,
                n_splits=self.signal_folds,
                random_state=self.random_state,
                smoothing=self.smoothing,
                min_support=self.min_support,
            )
        matrix = self._prepare_matrix(raw, train_signals)
        self.model_feature_names_ = matrix.columns.tolist()
        categorical_indices = [
            matrix.columns.get_loc(column) for column in CATEGORICAL_MODEL_COLUMNS
        ]
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            loss_function="Logloss",
            eval_metric="Accuracy",
            bootstrap_type="Bayesian",
            bagging_temperature=0.5,
            random_seed=self.random_state,
            allow_writing_files=False,
            thread_count=1,
            verbose=False,
        )
        self.model_.fit(matrix, labels, cat_features=categorical_indices)
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.n_features_in_ = len(RAW_FEATURE_COLUMNS)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self, ["feature_engineer_", "group_signal_map_", "model_"]
        )
        self._validate_raw(X)
        raw = X[RAW_FEATURE_COLUMNS].copy()
        signals = self.group_signal_map_.transform(raw) if self.use_group_signals else None
        matrix = self._prepare_matrix(raw, signals)
        return np.asarray(self.model_.predict_proba(matrix), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probability = self.predict_proba(X)[:, 1]
        return (probability >= self.decision_threshold).astype(int)


def make_catboost_baseline(**params: object) -> FamilyTicketCatBoostClassifier:
    """Return the same CatBoost feature model without label-derived signals."""

    return FamilyTicketCatBoostClassifier(use_group_signals=False, **params)
