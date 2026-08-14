"""Leak-safe family and ticket survival evidence for Titanic models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


GROUP_SIGNAL_COLUMNS = [
    "demographic_probability",
    "family_support",
    "family_probability",
    "family_delta",
    "family_confidence",
    "family_available",
    "ticket_support",
    "ticket_probability",
    "ticket_delta",
    "ticket_confidence",
    "ticket_available",
]


def _require_columns(frame: pd.DataFrame) -> None:
    required = {"PassengerId", "Name", "SibSp", "Parch", "Ticket", "Sex", "Pclass"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing Titanic group columns: {missing}")


def normalize_group_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable keys without allowing singleton surnames to share evidence."""

    _require_columns(frame)
    surname = (
        frame["Name"]
        .fillna("")
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    family_size = (
        frame["SibSp"].fillna(0).astype(int)
        + frame["Parch"].fillna(0).astype(int)
        + 1
    )
    passenger_token = frame["PassengerId"].astype(int).astype(str)
    shared_family = surname + "_" + family_size.astype(str)
    family_key = shared_family.where(
        family_size.gt(1), "__SINGLETON__" + passenger_token
    )

    ticket_key = (
        frame["Ticket"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    ticket_key = ticket_key.where(
        ticket_key.str.len().gt(0), "__MISSING_TICKET__" + passenger_token
    )
    return pd.DataFrame(
        {"FamilyKey": family_key, "TicketKey": ticket_key}, index=frame.index
    )


@dataclass
class GroupSignalMap:
    """Fit training-only survival maps and transform unseen passengers."""

    smoothing: float = 2.0
    min_support: int = 2

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "GroupSignalMap":
        _require_columns(X)
        labels = np.asarray(y, dtype=int)
        if len(labels) != len(X):
            raise ValueError("X and y must have the same number of rows")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("Survival labels must be binary")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")
        if self.min_support < 1:
            raise ValueError("min_support must be at least one")

        keys = normalize_group_keys(X)
        labelled = keys.assign(_target=labels)
        self.prior_probability_ = float(labels.mean())
        self.family_stats_ = labelled.groupby("FamilyKey")["_target"].agg(
            ["count", "sum"]
        )
        self.ticket_stats_ = labelled.groupby("TicketKey")["_target"].agg(
            ["count", "sum"]
        )

        demographics = pd.DataFrame(
            {
                "Sex": X["Sex"].fillna("unknown").astype(str),
                "Pclass": X["Pclass"].fillna(-1).astype(int),
                "_target": labels,
            },
            index=X.index,
        )
        demographic_stats = demographics.groupby(["Sex", "Pclass"])["_target"].agg(
            ["count", "sum"]
        )
        self.demographic_probability_ = {
            (str(sex), int(pclass)): float(
                (row["sum"] + self.smoothing * self.prior_probability_)
                / (row["count"] + self.smoothing)
            )
            for (sex, pclass), row in demographic_stats.iterrows()
        }
        return self

    def _demographic_prior(self, X: pd.DataFrame) -> pd.Series:
        if not hasattr(self, "prior_probability_"):
            raise ValueError("GroupSignalMap must be fitted before transform")
        values = [
            self.demographic_probability_.get(
                (str(sex), int(pclass)), self.prior_probability_
            )
            for sex, pclass in zip(
                X["Sex"].fillna("unknown"),
                X["Pclass"].fillna(-1),
                strict=True,
            )
        ]
        return pd.Series(values, index=X.index, dtype=float)

    def _transform_group(
        self,
        key: pd.Series,
        stats: pd.DataFrame,
        prefix: str,
        fallback: pd.Series,
    ) -> pd.DataFrame:
        support = key.map(stats["count"]).fillna(0).astype(float)
        survivors = key.map(stats["sum"]).fillna(0).astype(float)
        available = support.ge(self.min_support)
        smoothed = (
            survivors + self.smoothing * self.prior_probability_
        ) / (support + self.smoothing)
        probability = smoothed.where(available, fallback)
        used_support = support.where(available, 0.0)
        confidence = (support / (support + self.smoothing)).where(available, 0.0)
        return pd.DataFrame(
            {
                f"{prefix}_support": used_support,
                f"{prefix}_probability": probability,
                f"{prefix}_delta": probability - fallback,
                f"{prefix}_confidence": confidence,
                f"{prefix}_available": available.astype(float),
            },
            index=key.index,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _require_columns(X)
        keys = normalize_group_keys(X)
        demographic = self._demographic_prior(X)
        family = self._transform_group(
            keys["FamilyKey"], self.family_stats_, "family", demographic
        )
        ticket = self._transform_group(
            keys["TicketKey"], self.ticket_stats_, "ticket", demographic
        )
        result = pd.concat(
            [demographic.rename("demographic_probability"), family, ticket], axis=1
        )[GROUP_SIGNAL_COLUMNS]
        if result.isna().any().any() or not np.isfinite(result.to_numpy()).all():
            raise AssertionError("Group survival features must be finite and complete")
        return result


def cross_fit_group_signals(
    X: pd.DataFrame,
    y: Sequence[int],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    smoothing: float = 2.0,
    min_support: int = 2,
    splits: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Create one training-only group-signal row for every input row."""

    labels = np.asarray(y, dtype=int)
    if len(labels) != len(X):
        raise ValueError("X and y must have the same number of rows")
    if splits is None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        split_list = list(splitter.split(X, labels))
    else:
        split_list = list(splits)

    values = np.full((len(X), len(GROUP_SIGNAL_COLUMNS)), np.nan, dtype=float)
    coverage = np.zeros(len(X), dtype=int)
    for train_index, validation_index in split_list:
        train_index = np.asarray(train_index, dtype=int)
        validation_index = np.asarray(validation_index, dtype=int)
        mapping = GroupSignalMap(
            smoothing=smoothing, min_support=min_support
        ).fit(X.iloc[train_index], labels[train_index])
        transformed = mapping.transform(X.iloc[validation_index])
        values[validation_index] = transformed.to_numpy(dtype=float)
        coverage[validation_index] += 1

    if not np.all(coverage == 1) or np.isnan(values).any():
        raise AssertionError("Cross-fitting must cover every row exactly once")
    return pd.DataFrame(values, columns=GROUP_SIGNAL_COLUMNS, index=X.index)
