"""Nested evaluation and auditable outputs for family/ticket CatBoost."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.family_ticket_model import (
    FamilyTicketCatBoostClassifier,
    make_catboost_baseline,
)
from src.family_ticket_signal import normalize_group_keys
from titanic_b_pipeline import RAW_FEATURE_COLUMNS, validate_raw_data


REQUIRED_MANIFEST_FIELDS = {
    "status",
    "generated_at_utc",
    "source_hashes",
    "git_commit",
    "versions",
    "random_state",
    "selected_configuration",
    "fold_metrics",
    "evidence_coverage",
    "output_paths",
    "output_hashes",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def validate_submission(frame: pd.DataFrame, expected_ids: pd.Series) -> None:
    """Validate the exact Kaggle Titanic submission contract."""

    if frame.columns.tolist() != ["PassengerId", "Survived"]:
        raise ValueError("Submission columns must be PassengerId,Survived")
    expected = pd.Series(expected_ids, dtype=int).sort_values().reset_index(drop=True)
    if len(frame) != len(expected):
        raise ValueError("Submission row count does not match the test set")
    if frame.isna().any().any() or frame["PassengerId"].duplicated().any():
        raise ValueError("Submission IDs and predictions must be complete and unique")
    actual = frame["PassengerId"].astype(int).sort_values().reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Submission PassengerIds do not match the test set")
    if not set(frame["Survived"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("Submission predictions must be binary")
    if not np.equal(frame["Survived"], frame["Survived"].astype(int)).all():
        raise ValueError("Submission predictions must be integer-valued")


def validate_manifest(manifest: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(manifest))
    if missing:
        raise ValueError(f"Run manifest is missing fields: {missing}")
    if manifest["status"] != "PASS":
        raise ValueError("Run manifest status is not PASS")


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "predicted_survival_rate": float(prediction.mean()),
    }


def _model_params(quick: bool, random_state: int) -> dict[str, Any]:
    return {
        "iterations": 40 if quick else 260,
        "depth": 4 if quick else 5,
        "learning_rate": 0.08 if quick else 0.03,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "signal_folds": 2 if quick else 4,
        "random_state": random_state,
    }


def _candidate_grid(quick: bool) -> list[dict[str, Any]]:
    if quick:
        return [
            {
                "smoothing": 2.0,
                "min_support": 2,
                "decision_threshold": 0.50,
                "policy": "none",
            },
            {
                "smoothing": 2.0,
                "min_support": 2,
                "decision_threshold": 0.58,
                "policy": "none",
            },
        ]
    return [
        {
            "smoothing": 2.0,
            "min_support": 2,
            "decision_threshold": threshold,
            "policy": policy,
        }
        for threshold, policy in [
            (0.50, "none"),
            (0.54, "none"),
            (0.58, "none"),
            (0.50, "conservative"),
            (0.54, "conservative"),
            (0.58, "conservative"),
        ]
    ]


def _build_model(
    config: dict[str, Any], *, quick: bool, random_state: int, baseline: bool = False
) -> FamilyTicketCatBoostClassifier:
    params = _model_params(quick, random_state)
    params.update(
        {
            "smoothing": config["smoothing"],
            "min_support": config["min_support"],
            "decision_threshold": config["decision_threshold"],
        }
    )
    return make_catboost_baseline(**params) if baseline else FamilyTicketCatBoostClassifier(**params)


def _apply_policy(
    model: FamilyTicketCatBoostClassifier,
    X: pd.DataFrame,
    probability: np.ndarray,
    policy: str,
    decision_threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    adjusted = np.asarray(probability, dtype=float).copy()
    if policy == "none":
        return adjusted, {"family_low": 0, "ticket_high": 0, "changed": 0}
    if policy != "conservative":
        raise ValueError(f"Unknown candidate policy: {policy}")

    signals = model.group_signal_map_.transform(X)
    title = X["Name"].fillna("").str.extract(r",\s*([^.]*)\.", expand=False)
    child = X["Age"].fillna(99).lt(16) | title.eq("Master")
    vulnerable = X["Sex"].eq("female") | child
    adult_male = X["Sex"].eq("male") & ~child
    family_low = (
        vulnerable
        & signals["family_available"].eq(1)
        & signals["family_probability"].le(0.30)
        & signals["family_support"].ge(2)
    )
    ticket_high = (
        adult_male
        & signals["ticket_available"].eq(1)
        & signals["ticket_probability"].ge(0.65)
        & signals["ticket_support"].ge(2)
    )
    threshold = model.decision_threshold if decision_threshold is None else decision_threshold
    before = adjusted >= threshold
    adjusted[family_low.to_numpy()] = np.minimum(adjusted[family_low.to_numpy()], 0.20)
    adjusted[ticket_high.to_numpy()] = np.maximum(adjusted[ticket_high.to_numpy()], 0.80)
    after = adjusted >= threshold
    return adjusted, {
        "family_low": int(family_low.sum()),
        "ticket_high": int(ticket_high.sum()),
        "changed": int(np.not_equal(before, after).sum()),
    }


def _evaluate_configs(
    X: pd.DataFrame,
    y: np.ndarray,
    configs: list[dict[str, Any]],
    *,
    n_splits: int,
    quick: bool,
    random_state: int,
) -> list[dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    split_list = list(splitter.split(X, y))
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for config in configs:
        grouped.setdefault(
            (float(config["smoothing"]), int(config["min_support"])), []
        ).append(config)
    results: dict[str, dict[str, Any]] = {
        json.dumps(config, sort_keys=True): {
            "configuration": config,
            "fold_accuracies": [],
            "changed_rows": 0,
        }
        for config in configs
    }
    for model_group, model_configs in grouped.items():
        representative = dict(model_configs[0])
        representative["decision_threshold"] = 0.5
        representative["policy"] = "none"
        for fold, (train_index, validation_index) in enumerate(split_list, start=1):
            model = _build_model(
                representative, quick=quick, random_state=random_state + fold * 100
            ).fit(X.iloc[train_index], y[train_index])
            raw_probability = model.predict_proba(X.iloc[validation_index])[:, 1]
            for config in model_configs:
                probability, audit = _apply_policy(
                    model,
                    X.iloc[validation_index],
                    raw_probability,
                    config["policy"],
                    config["decision_threshold"],
                )
                record = results[json.dumps(config, sort_keys=True)]
                record["fold_accuracies"].append(
                    float(
                        accuracy_score(
                            y[validation_index],
                            probability >= config["decision_threshold"],
                        )
                    )
                )
                record["changed_rows"] += audit["changed"]
    finalized = []
    for config in configs:
        record = results[json.dumps(config, sort_keys=True)]
        scores = record["fold_accuracies"]
        finalized.append(
            {
                **record,
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores)),
                "changed_rows": int(record["changed_rows"]),
            }
        )
    return finalized


def _choose_config(results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        results,
        key=lambda item: (
            -round(item["mean_accuracy"], 12),
            round(item["std_accuracy"], 12),
            item["changed_rows"],
            json.dumps(item["configuration"], sort_keys=True),
        ),
    )
    return dict(ordered[0]["configuration"])


def _blend_weights() -> tuple[float, ...]:
    return (0.0, 0.25, 0.5, 0.75, 1.0)


def _choose_blend_weight(results: list[dict[str, Any]]) -> float:
    ordered = sorted(
        results,
        key=lambda item: (
            -round(item["mean_accuracy"], 12),
            item["family_weight"],
            round(item["std_accuracy"], 12),
        ),
    )
    return float(ordered[0]["family_weight"])


def _evaluate_blend_weights(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict[str, Any],
    *,
    n_splits: int,
    quick: bool,
    random_state: int,
) -> list[dict[str, Any]]:
    """Select a family-model probability weight using validation-only fits."""

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {
        weight: {"family_weight": weight, "fold_accuracies": []}
        for weight in _blend_weights()
    }
    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y), start=1):
        train_X, train_y = X.iloc[train_index], y[train_index]
        validation_X = X.iloc[validation_index]
        family = _build_model(
            config, quick=quick, random_state=random_state + fold * 100
        ).fit(train_X, train_y)
        family_probability, _ = _apply_policy(
            family,
            validation_X,
            family.predict_proba(validation_X)[:, 1],
            config["policy"],
            0.50,
        )
        baseline = _build_model(
            config,
            quick=quick,
            random_state=random_state + fold * 100,
            baseline=True,
        ).fit(train_X, train_y)
        baseline_probability = baseline.predict_proba(validation_X)[:, 1]
        for weight, record in results.items():
            blend_probability = (
                weight * family_probability + (1.0 - weight) * baseline_probability
            )
            record["fold_accuracies"].append(
                float(accuracy_score(y[validation_index], blend_probability >= 0.50))
            )
    return [
        {
            **record,
            "mean_accuracy": float(np.mean(record["fold_accuracies"])),
            "std_accuracy": float(np.std(record["fold_accuracies"])),
        }
        for record in results.values()
    ]


def _connected_groups(X: pd.DataFrame) -> np.ndarray:
    keys = normalize_group_keys(X).reset_index(drop=True)
    parent = list(range(len(keys)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for column in ("FamilyKey", "TicketKey"):
        first: dict[str, int] = {}
        for index, key in enumerate(keys[column].astype(str)):
            if key in first:
                union(first[key], index)
            else:
                first[key] = index
    roots = [find(index) for index in range(len(keys))]
    return pd.factorize(np.asarray(roots, dtype=int), sort=True)[0]


def _grouped_robustness(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict[str, Any],
    *,
    quick: bool,
    random_state: int,
) -> dict[str, float]:
    folds = 3 if quick else 5
    groups = _connected_groups(X)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=random_state
    )
    signal_probability = np.full(len(X), np.nan)
    baseline_probability = np.full(len(X), np.nan)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        signal = _build_model(
            config, quick=quick, random_state=random_state + fold * 1000
        ).fit(X.iloc[train_index], y[train_index])
        raw_probability = signal.predict_proba(X.iloc[validation_index])[:, 1]
        signal_probability[validation_index], _ = _apply_policy(
            signal, X.iloc[validation_index], raw_probability, config["policy"]
        )
        baseline = _build_model(
            config,
            quick=quick,
            random_state=random_state + fold * 1000,
            baseline=True,
        ).fit(X.iloc[train_index], y[train_index])
        baseline_probability[validation_index] = baseline.predict_proba(
            X.iloc[validation_index]
        )[:, 1]
    return {
        "signal_accuracy": float(
            accuracy_score(y, signal_probability >= config["decision_threshold"])
        ),
        "baseline_accuracy": float(accuracy_score(y, baseline_probability >= 0.5)),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run_experiment(
    train_path: str | Path,
    test_path: str | Path,
    output_root: str | Path,
    *,
    baseline_submission_path: str | Path | None = None,
    quick: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    train_path, test_path = Path(train_path).resolve(), Path(test_path).resolve()
    output_root = Path(output_root).resolve()
    reports = output_root / "reports" / "family_ticket"
    outputs = output_root / "outputs"
    models = output_root / "models"
    for directory in (reports, outputs, models):
        directory.mkdir(parents=True, exist_ok=True)

    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    validate_raw_data(train, test)
    X = train[RAW_FEATURE_COLUMNS].copy()
    y = train["Survived"].astype(int).to_numpy()
    X_test = test[RAW_FEATURE_COLUMNS].copy()
    outer_folds, inner_folds = (3, 2) if quick else (5, 3)
    outer = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=random_state
    )
    signal_oof = np.full(len(X), np.nan)
    baseline_oof = np.full(len(X), np.nan)
    blend_oof = np.full(len(X), np.nan)
    signal_oof_prediction = np.zeros(len(X), dtype=int)
    baseline_oof_prediction = np.zeros(len(X), dtype=int)
    blend_oof_prediction = np.zeros(len(X), dtype=int)
    selected_blend_weight = np.full(len(X), np.nan)
    outer_fold = np.zeros(len(X), dtype=int)
    fold_metrics: list[dict[str, Any]] = []
    nested_config_results: list[dict[str, Any]] = []
    nested_blend_results: list[dict[str, Any]] = []

    for fold, (train_index, validation_index) in enumerate(outer.split(X, y), start=1):
        inner_X, inner_y = X.iloc[train_index], y[train_index]
        config_results = _evaluate_configs(
            inner_X,
            inner_y,
            _candidate_grid(quick),
            n_splits=inner_folds,
            quick=quick,
            random_state=random_state + fold * 1000,
        )
        selected = _choose_config(config_results)
        for result in config_results:
            nested_config_results.append({"outer_fold": fold, **result})
        blend_results = _evaluate_blend_weights(
            inner_X,
            inner_y,
            selected,
            n_splits=inner_folds,
            quick=quick,
            random_state=random_state + fold * 2000,
        )
        family_weight = _choose_blend_weight(blend_results)
        for result in blend_results:
            nested_blend_results.append({"outer_fold": fold, **result})

        signal = _build_model(
            selected, quick=quick, random_state=random_state + fold * 10000
        ).fit(inner_X, inner_y)
        raw_probability = signal.predict_proba(X.iloc[validation_index])[:, 1]
        probability, policy_audit = _apply_policy(
            signal, X.iloc[validation_index], raw_probability, selected["policy"]
        )
        signal_oof[validation_index] = probability
        signal_oof_prediction[validation_index] = (
            probability >= selected["decision_threshold"]
        ).astype(int)

        baseline = _build_model(
            selected,
            quick=quick,
            random_state=random_state + fold * 10000,
            baseline=True,
        ).fit(inner_X, inner_y)
        baseline_oof[validation_index] = baseline.predict_proba(
            X.iloc[validation_index]
        )[:, 1]
        baseline_oof_prediction[validation_index] = (
            baseline_oof[validation_index] >= 0.5
        ).astype(int)
        blend_oof[validation_index] = (
            family_weight * probability + (1.0 - family_weight) * baseline_oof[validation_index]
        )
        blend_oof_prediction[validation_index] = (
            blend_oof[validation_index] >= 0.50
        ).astype(int)
        selected_blend_weight[validation_index] = family_weight
        outer_fold[validation_index] = fold
        signal_metrics = _metrics(
            y[validation_index], probability, selected["decision_threshold"]
        )
        baseline_metrics = _metrics(y[validation_index], baseline_oof[validation_index], 0.5)
        fold_metrics.append(
            {
                "fold": fold,
                "selected_configuration": selected,
                "signal_accuracy": signal_metrics["accuracy"],
                "signal_f1": signal_metrics["f1"],
                "signal_roc_auc": signal_metrics["roc_auc"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "baseline_f1": baseline_metrics["f1"],
                "baseline_roc_auc": baseline_metrics["roc_auc"],
                "blend_family_weight": family_weight,
                "blend_accuracy": float(
                    accuracy_score(y[validation_index], blend_oof_prediction[validation_index])
                ),
                **{f"policy_{key}": value for key, value in policy_audit.items()},
            }
        )

    if (
        np.isnan(signal_oof).any()
        or np.isnan(baseline_oof).any()
        or np.isnan(blend_oof).any()
        or np.isnan(selected_blend_weight).any()
        or (outer_fold == 0).any()
    ):
        raise AssertionError("Outer OOF predictions must cover all training rows")

    full_config_results = _evaluate_configs(
        X,
        y,
        _candidate_grid(quick),
        n_splits=inner_folds,
        quick=quick,
        random_state=random_state + 90000,
    )
    selected = _choose_config(full_config_results)
    full_blend_results = _evaluate_blend_weights(
        X,
        y,
        selected,
        n_splits=inner_folds,
        quick=quick,
        random_state=random_state + 95000,
    )
    deployment_family_weight = _choose_blend_weight(full_blend_results)
    deployment_model = _build_model(
        selected, quick=quick, random_state=random_state + 100000
    ).fit(X, y)
    test_probability = deployment_model.predict_proba(X_test)[:, 1]
    test_probability, deployment_policy = _apply_policy(
        deployment_model, X_test, test_probability, selected["policy"]
    )
    primary = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": (
                test_probability >= selected["decision_threshold"]
            ).astype(int),
        }
    )
    validate_submission(primary, test["PassengerId"])

    deployment_baseline_model = _build_model(
        selected, quick=quick, random_state=random_state + 100000, baseline=True
    ).fit(X, y)
    baseline_test_probability = deployment_baseline_model.predict_proba(X_test)[:, 1]
    blend = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": (
                deployment_family_weight * test_probability
                + (1.0 - deployment_family_weight) * baseline_test_probability
                >= 0.50
            ).astype(int),
        }
    )
    validate_submission(blend, test["PassengerId"])

    primary_path = outputs / "submission_family_ticket_catboost.csv"
    baseline_path = outputs / "submission_catboost_baseline.csv"
    blend_path = outputs / "submission_catboost_family_blend.csv"
    primary.to_csv(primary_path, index=False)
    blend.to_csv(blend_path, index=False)
    if baseline_submission_path is not None:
        baseline_submission = pd.read_csv(Path(baseline_submission_path))
        validate_submission(baseline_submission, test["PassengerId"])
        baseline_submission = baseline_submission[["PassengerId", "Survived"]].astype(int)
    else:
        baseline_submission = pd.DataFrame(
            {
                "PassengerId": test["PassengerId"].astype(int),
                "Survived": (baseline_test_probability >= 0.50).astype(int),
            }
        )
    baseline_submission.to_csv(baseline_path, index=False)

    model_path = models / "family_ticket_catboost.joblib"
    joblib.dump(
        {
            "model": deployment_model,
            "baseline_model": deployment_baseline_model,
            "configuration": selected,
            "policy": selected["policy"],
            "selected_family_weight": deployment_family_weight,
            "blend_threshold": 0.50,
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
        },
        model_path,
    )

    comparison = {
        "signal_accuracy": float(
            accuracy_score(y, signal_oof_prediction)
        ),
        "baseline_accuracy": float(accuracy_score(y, baseline_oof_prediction)),
        "blend_accuracy": float(accuracy_score(y, blend_oof_prediction)),
        "signal_f1": float(f1_score(y, signal_oof_prediction)),
        "baseline_f1": float(f1_score(y, baseline_oof_prediction)),
        "blend_f1": float(f1_score(y, blend_oof_prediction)),
        "signal_roc_auc": float(roc_auc_score(y, signal_oof)),
        "baseline_roc_auc": float(roc_auc_score(y, baseline_oof)),
        "blend_roc_auc": float(roc_auc_score(y, blend_oof)),
    }
    family_promotion_status = (
        "promoted"
        if comparison["signal_accuracy"] > comparison["baseline_accuracy"]
        else "baseline_retained"
    )
    blend_promotion_status = (
        "promoted"
        if comparison["blend_accuracy"] > comparison["baseline_accuracy"]
        else "baseline_retained"
    )
    promotion_status = blend_promotion_status
    recommended = blend if blend_promotion_status == "promoted" else baseline_submission
    recommended_path = outputs / "submission_recommended.csv"
    validate_submission(recommended, test["PassengerId"])
    recommended.to_csv(recommended_path, index=False)
    grouped = _grouped_robustness(
        X, y, selected, quick=quick, random_state=random_state + 200000
    )
    full_signals = deployment_model.group_signal_map_.transform(X_test)
    evidence_coverage = {
        "family_rows": int(full_signals["family_available"].sum()),
        "ticket_rows": int(full_signals["ticket_available"].sum()),
        "policy_family_low": deployment_policy["family_low"],
        "policy_ticket_high": deployment_policy["ticket_high"],
        "policy_changed": deployment_policy["changed"],
    }

    oof = pd.DataFrame(
        {
            "PassengerId": train["PassengerId"].astype(int),
            "Survived": y,
            "OuterFold": outer_fold,
            "SignalProbability": signal_oof,
            "SignalPrediction": signal_oof_prediction,
            "BaselineProbability": baseline_oof,
            "BaselinePrediction": baseline_oof_prediction,
            "BlendProbability": blend_oof,
            "BlendPrediction": blend_oof_prediction,
            "BlendFamilyWeight": selected_blend_weight,
        }
    )
    oof_path = reports / "oof_predictions.csv"
    folds_path = reports / "outer_fold_metrics.csv"
    comparison_path = reports / "model_comparison.csv"
    config_path = reports / "configuration_results.csv"
    blend_config_path = reports / "blend_configuration_results.csv"
    difference_path = reports / "submission_difference.csv"
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(fold_metrics).to_csv(folds_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "family_ticket_catboost",
                "accuracy": comparison["signal_accuracy"],
                "f1": comparison["signal_f1"],
                "roc_auc": comparison["signal_roc_auc"],
            },
            {
                "model": "same_split_catboost_baseline",
                "accuracy": comparison["baseline_accuracy"],
                "f1": comparison["baseline_f1"],
                "roc_auc": comparison["baseline_roc_auc"],
            },
            {
                "model": "nested_catboost_family_blend",
                "accuracy": comparison["blend_accuracy"],
                "f1": comparison["blend_f1"],
                "roc_auc": comparison["blend_roc_auc"],
            },
        ]
    ).to_csv(comparison_path, index=False)
    pd.DataFrame(
        [
            {
                "scope": scope,
                "outer_fold": fold,
                **result["configuration"],
                "mean_accuracy": result["mean_accuracy"],
                "std_accuracy": result["std_accuracy"],
                "changed_rows": result["changed_rows"],
            }
            for scope, fold, result in [
                *[
                    ("nested_outer_selection", item["outer_fold"], item)
                    for item in nested_config_results
                ],
                *[
                    ("full_training_selection", 0, item)
                    for item in full_config_results
                ],
            ]
        ]
    ).to_csv(config_path, index=False)
    pd.DataFrame(
        [
            {
                "scope": scope,
                "outer_fold": fold,
                "family_weight": result["family_weight"],
                "mean_accuracy": result["mean_accuracy"],
                "std_accuracy": result["std_accuracy"],
                "fold_accuracies": json.dumps(result["fold_accuracies"]),
            }
            for scope, fold, result in [
                *[
                    ("nested_outer_selection", item["outer_fold"], item)
                    for item in nested_blend_results
                ],
                *[
                    ("full_training_selection", 0, item)
                    for item in full_blend_results
                ],
            ]
        ]
    ).to_csv(blend_config_path, index=False)

    difference = primary.merge(
        baseline_submission, on="PassengerId", suffixes=("_new", "_baseline")
    )
    difference["Changed"] = difference["Survived_new"].ne(
        difference["Survived_baseline"]
    )
    difference.to_csv(difference_path, index=False)

    overview_path = reports / "overview.md"
    overview_path.write_text(
        "\n".join(
            [
                "# Family/Ticket CatBoost Experiment",
                "",
                "This run uses only the official Kaggle Titanic train/test files and training labels.",
                "",
                f"- Same-split signal OOF accuracy: `{comparison['signal_accuracy']:.4f}`",
                f"- Same-split CatBoost baseline OOF accuracy: `{comparison['baseline_accuracy']:.4f}`",
                f"- Nested blend OOF accuracy: `{comparison['blend_accuracy']:.4f}`",
                f"- Deployment family blend weight: `{deployment_family_weight:.2f}`",
                f"- Group-disjoint signal accuracy: `{grouped['signal_accuracy']:.4f}`",
                f"- Group-disjoint baseline accuracy: `{grouped['baseline_accuracy']:.4f}`",
                f"- Uploaded CatBoost Kaggle baseline: `0.79186`",
                f"- Test predictions changed from uploaded baseline: `{int(difference['Changed'].sum())}`",
                f"- Selected configuration: `{json.dumps(selected, sort_keys=True)}`",
                f"- Blend promotion decision: `{blend_promotion_status}`",
                f"- Recommended Kaggle file: `outputs/{recommended_path.name}`",
                "",
                "The 0.83 Kaggle target remains unverified until the recommended CSV is uploaded.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_paths = {
        "submission": str(primary_path.relative_to(output_root)),
        "blend_submission": str(blend_path.relative_to(output_root)),
        "recommended_submission": str(recommended_path.relative_to(output_root)),
        "baseline_submission": str(baseline_path.relative_to(output_root)),
        "model": str(model_path.relative_to(output_root)),
        "oof": str(oof_path.relative_to(output_root)),
        "fold_metrics": str(folds_path.relative_to(output_root)),
        "comparison": str(comparison_path.relative_to(output_root)),
        "configuration_results": str(config_path.relative_to(output_root)),
        "blend_configuration_results": str(blend_config_path.relative_to(output_root)),
        "difference_audit": str(difference_path.relative_to(output_root)),
        "overview": str(overview_path.relative_to(output_root)),
    }
    output_hashes = {
        name: _sha256(output_root / path) for name, path in output_paths.items()
    }
    manifest: dict[str, Any] = {
        "status": "PASS",
        "mode": "quick" if quick else "full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {"train": _sha256(train_path), "test": _sha256(test_path)},
        "git_commit": _git_commit(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "joblib": joblib.__version__,
        },
        "random_state": random_state,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "selected_configuration": selected,
        "promotion_status": promotion_status,
        "blend_promotion_status": blend_promotion_status,
        "family_promotion_status": family_promotion_status,
        "blend": {
            "threshold": 0.50,
            "candidate_family_weights": list(_blend_weights()),
            "selected_family_weight": deployment_family_weight,
            "outer_selected_family_weights": [
                metric["blend_family_weight"] for metric in fold_metrics
            ],
            "outer_oof_comparison": {
                "blend_accuracy": comparison["blend_accuracy"],
                "baseline_accuracy": comparison["baseline_accuracy"],
            },
        },
        "fold_metrics": fold_metrics,
        "comparison": comparison,
        "grouped_robustness": grouped,
        "evidence_coverage": evidence_coverage,
        "oof_rows": len(oof),
        "predicted_survivors": int(primary["Survived"].sum()),
        "recommended_predicted_survivors": int(recommended["Survived"].sum()),
        "baseline_difference_rows": int(difference["Changed"].sum()),
        "output_paths": output_paths,
        "output_hashes": output_hashes,
    }
    manifest = _json_ready(manifest)
    validate_manifest(manifest)
    manifest_path = reports / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_path = reports / "experiment_index.jsonl"
    index_entry = {
        "generated_at_utc": manifest["generated_at_utc"],
        "mode": manifest["mode"],
        "git_commit": manifest["git_commit"],
        "selected_configuration": selected,
        "comparison": comparison,
        "promotion_status": promotion_status,
        "blend_promotion_status": blend_promotion_status,
        "selected_family_weight": deployment_family_weight,
        "submission_sha256": output_hashes["recommended_submission"],
    }
    with index_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(index_entry, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest
