from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.family_ticket_experiment import (
    _blend_weights,
    _choose_blend_weight,
    run_experiment,
    validate_manifest,
    validate_submission,
)


ROOT = Path(__file__).resolve().parents[1]


class FamilyTicketExperimentTests(unittest.TestCase):
    def test_blend_weights_are_fixed_and_include_both_controls(self) -> None:
        self.assertEqual(_blend_weights(), (0.0, 0.25, 0.5, 0.75, 1.0))

    def test_blend_selection_prefers_lower_family_weight_on_accuracy_tie(self) -> None:
        results = [
            {"family_weight": 0.50, "mean_accuracy": 0.82, "std_accuracy": 0.01},
            {"family_weight": 0.25, "mean_accuracy": 0.82, "std_accuracy": 0.03},
        ]
        self.assertEqual(_choose_blend_weight(results), 0.25)

    def test_submission_validator_rejects_invalid_contracts(self) -> None:
        expected = pd.Series([892, 893, 894], dtype=int)
        valid = pd.DataFrame(
            {"PassengerId": [892, 893, 894], "Survived": [0, 1, 0]}
        )
        validate_submission(valid, expected)

        duplicate = valid.copy()
        duplicate.loc[2, "PassengerId"] = 893
        with self.assertRaises(ValueError):
            validate_submission(duplicate, expected)

        non_binary = valid.copy()
        non_binary.loc[1, "Survived"] = 2
        with self.assertRaises(ValueError):
            validate_submission(non_binary, expected)

    def test_manifest_validator_requires_auditable_fields(self) -> None:
        manifest = {
            "status": "PASS",
            "generated_at_utc": "2026-08-14T00:00:00+00:00",
            "source_hashes": {"train": "a", "test": "b"},
            "git_commit": "abc123",
            "versions": {"python": "3.12"},
            "random_state": 42,
            "selected_configuration": {"smoothing": 2.0, "min_support": 2},
            "fold_metrics": [{"fold": 1, "accuracy": 0.8}],
            "evidence_coverage": {"family_rows": 1, "ticket_rows": 2},
            "output_paths": {"submission": "outputs/submission.csv"},
            "output_hashes": {"submission": "hash"},
        }
        validate_manifest(manifest)
        del manifest["source_hashes"]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_quick_run_writes_complete_audit(self) -> None:
        test = pd.read_csv(ROOT / "source" / "test.csv")
        baseline = pd.DataFrame(
            {
                "PassengerId": test["PassengerId"].astype(int),
                "Survived": test["Sex"].eq("female").astype(int),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            baseline_path = output / "input_baseline.csv"
            baseline.to_csv(baseline_path, index=False)
            summary = run_experiment(
                ROOT / "source" / "train.csv",
                ROOT / "source" / "test.csv",
                output,
                baseline_submission_path=baseline_path,
                quick=True,
            )

            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["outer_folds"], 3)
            self.assertEqual(len(summary["fold_metrics"]), 3)
            self.assertEqual(summary["oof_rows"], 891)
            self.assertIn("signal_accuracy", summary["comparison"])
            self.assertIn("baseline_accuracy", summary["comparison"])

            primary = pd.read_csv(
                output / "outputs" / "submission_family_ticket_catboost.csv"
            )
            validate_submission(primary, test["PassengerId"])
            recommended = pd.read_csv(
                output / "outputs" / "submission_recommended.csv"
            )
            validate_submission(recommended, test["PassengerId"])
            self.assertTrue((output / "models" / "family_ticket_catboost.joblib").is_file())
            manifest_path = output / "reports" / "family_ticket" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_manifest(manifest)
            self.assertIn(manifest["promotion_status"], {"promoted", "baseline_retained"})
            self.assertIn("recommended_submission", manifest["output_paths"])
            index_path = output / "reports" / "family_ticket" / "experiment_index.jsonl"
            self.assertEqual(len(index_path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
