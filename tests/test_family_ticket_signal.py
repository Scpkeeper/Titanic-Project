from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.family_ticket_signal import (
    GroupSignalMap,
    cross_fit_group_signals,
    normalize_group_keys,
)


def passenger_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6],
            "Name": [
                "Smith, Mrs. Ada",
                "Jones, Mr. Ben",
                "Smith, Miss. Cora",
                "Jones, Mrs. Dora",
                "Solo, Mr. Evan",
                "Solo, Miss. Fay",
            ],
            "SibSp": [1, 1, 1, 1, 0, 0],
            "Parch": [0, 0, 0, 0, 0, 0],
            "Ticket": [" A 1 ", "PC 2", "A1", "PC   2", "Z9", "Z9"],
            "Sex": ["female", "male", "female", "female", "male", "female"],
            "Pclass": [1, 2, 1, 2, 3, 3],
        }
    )


class GroupSignalTests(unittest.TestCase):
    def test_keys_normalize_and_singletons_never_share_family_evidence(self) -> None:
        keys = normalize_group_keys(passenger_fixture())
        self.assertEqual(keys.loc[0, "FamilyKey"], "SMITH_2")
        self.assertEqual(keys.loc[2, "FamilyKey"], "SMITH_2")
        self.assertEqual(keys.loc[0, "TicketKey"], "A1")
        self.assertEqual(keys.loc[2, "TicketKey"], "A1")
        self.assertNotEqual(keys.loc[4, "FamilyKey"], keys.loc[5, "FamilyKey"])

    def test_known_groups_receive_smoothed_evidence_and_unknowns_use_prior(self) -> None:
        frame = passenger_fixture().iloc[:4].copy()
        labels = np.asarray([1, 0, 1, 0], dtype=int)
        fitted = GroupSignalMap(smoothing=2.0, min_support=2).fit(frame, labels)

        known = fitted.transform(frame.iloc[[0]])
        self.assertEqual(float(known.iloc[0]["family_support"]), 2.0)
        self.assertEqual(float(known.iloc[0]["family_probability"]), 0.75)
        self.assertEqual(float(known.iloc[0]["family_confidence"]), 0.5)
        self.assertEqual(float(known.iloc[0]["family_available"]), 1.0)

        unknown = passenger_fixture().iloc[[4]].copy()
        transformed = fitted.transform(unknown)
        self.assertEqual(float(transformed.iloc[0]["family_support"]), 0.0)
        self.assertEqual(float(transformed.iloc[0]["family_probability"]), 0.5)
        self.assertEqual(float(transformed.iloc[0]["family_delta"]), 0.0)
        self.assertEqual(float(transformed.iloc[0]["family_available"]), 0.0)

    def test_cross_fit_excludes_own_label(self) -> None:
        frame = passenger_fixture()
        labels = np.asarray([1, 0, 1, 0, 0, 1], dtype=int)
        splits = [
            (np.asarray([2, 3, 4, 5]), np.asarray([0, 1])),
            (np.asarray([0, 1, 4, 5]), np.asarray([2, 3])),
            (np.asarray([0, 1, 2, 3]), np.asarray([4, 5])),
        ]
        original = cross_fit_group_signals(
            frame,
            labels,
            n_splits=3,
            random_state=42,
            smoothing=1.0,
            min_support=1,
            splits=splits,
        )
        changed_labels = labels.copy()
        changed_labels[0] = 0
        changed = cross_fit_group_signals(
            frame,
            changed_labels,
            n_splits=3,
            random_state=42,
            smoothing=1.0,
            min_support=1,
            splits=splits,
        )

        pd.testing.assert_series_equal(original.loc[0], changed.loc[0])
        self.assertNotEqual(
            float(original.loc[2, "family_probability"]),
            float(changed.loc[2, "family_probability"]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
