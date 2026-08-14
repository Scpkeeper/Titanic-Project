#!/usr/bin/env python3
"""Run the official-data-only family/ticket CatBoost experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.family_ticket_experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Cross-fitted family/ticket survival evidence with CatBoost"
    )
    parser.add_argument("--train", type=Path, default=root / "source" / "train.csv")
    parser.add_argument("--test", type=Path, default=root / "source" / "test.csv")
    parser.add_argument("--output", type=Path, default=root)
    parser.add_argument("--baseline-submission", type=Path)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_experiment(
        args.train,
        args.test,
        args.output,
        baseline_submission_path=args.baseline_submission,
        quick=args.quick,
    )
    print("Family/ticket CatBoost experiment complete")
    print(
        "Same-split OOF accuracy: "
        f"signal={result['comparison']['signal_accuracy']:.4f}; "
        f"baseline={result['comparison']['baseline_accuracy']:.4f}"
    )
    print(
        "Selected configuration: "
        + json.dumps(result["selected_configuration"], sort_keys=True)
    )
    print(f"Changed test predictions: {result['baseline_difference_rows']}")
    print(f"Primary submission: {result['output_paths']['submission']}")
    print("Kaggle target 0.83 is unverified until this CSV is submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
