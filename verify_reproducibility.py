#!/usr/bin/env python3
"""Compare two independently generated Titanic B output directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DETERMINISTIC_ARTIFACTS = [
    "data/Titanic_train_clean.csv",
    "data/Titanic_test_clean.csv",
    "data/Titanic_train_model_ready.csv",
    "data/Titanic_test_model_ready.csv",
    "reports/nested_cv_results.csv",
    "reports/feature_selection_results.csv",
    "reports/feature_scores.csv",
    "reports/missing_values_before_after.csv",
    "reports/field_dictionary.csv",
    "reports/cleaning_rules.csv",
    "reports/selected_features.txt",
    "reports/preprocessing_parameters.json",
    "reports/oof_predictions.json",
    "reports/quality_report.json",
    "reports/feature_selection_cv.png",
    "models/titanic_b_transform_selector.joblib",
    "models/titanic_b_logistic_baseline.joblib",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    reference = Path(args.reference)
    candidate = Path(args.candidate)
    records = []
    for relative in DETERMINISTIC_ARTIFACTS:
        reference_path = reference / relative
        candidate_path = candidate / relative
        reference_hash = sha256(reference_path) if reference_path.exists() else None
        candidate_hash = sha256(candidate_path) if candidate_path.exists() else None
        records.append(
            {
                "artifact": relative,
                "reference_sha256": reference_hash,
                "candidate_sha256": candidate_hash,
                "match": (
                    reference_hash is not None
                    and reference_hash == candidate_hash
                ),
            }
        )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["match"] for row in records) else "FAIL",
        "artifacts_checked": len(records),
        "artifacts_matched": sum(row["match"] for row in records),
        "comparison": records,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{report['status']}: "
        f"{report['artifacts_matched']}/{report['artifacts_checked']} "
        "deterministic artifacts matched"
    )
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
