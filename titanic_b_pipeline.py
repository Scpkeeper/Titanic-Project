#!/usr/bin/env python3
"""Titanic 模块 B：无泄漏数据清洗、特征工程、编码、缩放与特征选择。

正式交叉验证必须直接向 ``make_model_pipeline`` 传入原始 Kaggle 数据。
流水线中的每个有学习行为的步骤都会在交叉验证训练折内重新拟合。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


RANDOM_STATE = 42
TICKET_PREFIX_MIN_COUNT = 10

RAW_TRAIN_COLUMNS = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
RAW_FEATURE_COLUMNS = [column for column in RAW_TRAIN_COLUMNS if column != "Survived"]

SCALED_NUMERIC_COLUMNS = ["AgeFilled", "FareFilled"]
PASSTHROUGH_NUMERIC_COLUMNS = [
    "SexEncoded",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "FareLog",
    "FarePerPerson",
    "HasCabin",
    "AgePclass",
    "NameLength",
    "WomenChildPriority",
    "AgeMissing",
    "FareMissing",
    "CabinMissing",
]
CATEGORICAL_COLUMNS = [
    "Pclass",
    "Sex",
    "EmbarkedFilled",
    "Title",
    "CabinDeck",
    "TicketPrefix",
    "AgeBand",
    "FareBand",
    "FamilyGroup",
    "SexPclass",
]

TITLE_REPLACEMENTS = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = {"Mr", "Miss", "Mrs", "Master"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.columns.tolist() != RAW_TRAIN_COLUMNS:
        raise ValueError(
            "train.csv 字段不符合 Kaggle Titanic 标准格式："
            f"\n实际：{train.columns.tolist()}\n预期：{RAW_TRAIN_COLUMNS}"
        )
    if test.columns.tolist() != RAW_FEATURE_COLUMNS:
        raise ValueError(
            "test.csv 字段不符合 Kaggle Titanic 标准格式："
            f"\n实际：{test.columns.tolist()}\n预期：{RAW_FEATURE_COLUMNS}"
        )
    if train.shape != (891, 12) or test.shape != (418, 11):
        raise ValueError(f"数据形状异常：train={train.shape}, test={test.shape}")
    if train["PassengerId"].duplicated().any() or test["PassengerId"].duplicated().any():
        raise ValueError("PassengerId 存在重复值")
    if set(train["PassengerId"]) & set(test["PassengerId"]):
        raise ValueError("训练集与测试集 PassengerId 存在重叠")
    if not set(train["Survived"].unique()).issubset({0, 1}):
        raise ValueError("Survived 必须为 0/1")


def _extract_title(name: pd.Series) -> pd.Series:
    title = name.astype(str).str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    title = title.replace(TITLE_REPLACEMENTS)
    return title.where(title.isin(COMMON_TITLES), "Rare")


def _extract_ticket_prefix(ticket: pd.Series) -> pd.Series:
    def normalize(value: Any) -> str:
        cleaned = re.sub(r"[\d./\s]", "", str(value)).upper()
        return cleaned if cleaned else "NONE"

    return ticket.map(normalize)


def _family_group(size: pd.Series) -> pd.Series:
    return pd.cut(
        size,
        bins=[0, 1, 4, 6, np.inf],
        labels=["Solo", "Small", "Medium", "Large"],
        right=True,
    ).astype(str)


def _mutual_info_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """在 ``fit`` 中只学习训练折统计量，在 ``transform`` 中应用规则。"""

    def __init__(self, ticket_prefix_min_count: int = TICKET_PREFIX_MIN_COUNT):
        self.ticket_prefix_min_count = ticket_prefix_min_count

    @staticmethod
    def _validate_features(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TitanicFeatureEngineer 需要 pandas DataFrame")
        missing = set(RAW_FEATURE_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"缺少原始字段：{sorted(missing)}")

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TitanicFeatureEngineer":
        self._validate_features(X)
        data = X[RAW_FEATURE_COLUMNS].copy()
        data["Title"] = _extract_title(data["Name"])

        observed_age = data.dropna(subset=["Age"])
        self.age_group_medians_ = (
            observed_age.groupby(["Title", "Pclass", "Sex"], observed=True)["Age"]
            .median()
            .to_dict()
        )
        self.age_title_pclass_medians_ = (
            observed_age.groupby(["Title", "Pclass"], observed=True)["Age"]
            .median()
            .to_dict()
        )
        self.age_pclass_medians_ = (
            observed_age.groupby("Pclass", observed=True)["Age"].median().to_dict()
        )
        self.age_overall_median_ = float(observed_age["Age"].median())

        self.fare_pclass_medians_ = (
            data.groupby("Pclass", observed=True)["Fare"].median().to_dict()
        )
        self.fare_overall_median_ = float(data["Fare"].median())
        self.embarked_mode_ = str(data["Embarked"].mode(dropna=True).iloc[0])

        raw_prefix = _extract_ticket_prefix(data["Ticket"])
        prefix_counts = raw_prefix.value_counts()
        self.frequent_ticket_prefixes_ = sorted(
            prefix_counts[
                prefix_counts >= int(self.ticket_prefix_min_count)
            ].index.tolist()
        )

        fare_filled = data.apply(self._impute_fare_row, axis=1)
        _, fare_edges = pd.qcut(
            fare_filled,
            q=4,
            retbins=True,
            duplicates="drop",
        )
        fare_edges = np.asarray(fare_edges, dtype=float)
        fare_edges[0] = -np.inf
        fare_edges[-1] = np.inf
        self.fare_band_edges_ = fare_edges
        self.fare_band_labels_ = [
            f"Q{index + 1}" for index in range(len(fare_edges) - 1)
        ]
        self.feature_names_in_ = np.asarray(RAW_FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(RAW_FEATURE_COLUMNS)
        return self

    def _impute_age_row(self, row: pd.Series) -> float:
        if pd.notna(row["Age"]):
            return float(row["Age"])
        key = (str(row["Title"]), int(row["Pclass"]), str(row["Sex"]))
        title_pclass_key = (str(row["Title"]), int(row["Pclass"]))
        return float(
            self.age_group_medians_.get(
                key,
                self.age_title_pclass_medians_.get(
                    title_pclass_key,
                    self.age_pclass_medians_.get(
                        int(row["Pclass"]),
                        self.age_overall_median_,
                    ),
                ),
            )
        )

    def _impute_fare_row(self, row: pd.Series) -> float:
        if pd.notna(row["Fare"]):
            return float(row["Fare"])
        return float(
            self.fare_pclass_medians_.get(
                int(row["Pclass"]),
                self.fare_overall_median_,
            )
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            [
                "age_group_medians_",
                "fare_pclass_medians_",
                "embarked_mode_",
                "frequent_ticket_prefixes_",
                "fare_band_edges_",
            ],
        )
        self._validate_features(X)
        raw = X[RAW_FEATURE_COLUMNS].copy()
        data = pd.DataFrame(index=raw.index)

        data["PassengerId"] = raw["PassengerId"].astype("int64")
        data["Pclass"] = raw["Pclass"].astype("int8")
        data["Sex"] = raw["Sex"].astype(str)
        data["SexEncoded"] = raw["Sex"].map({"male": 0, "female": 1}).fillna(-1).astype("int8")

        data["Title"] = _extract_title(raw["Name"])
        data["AgeMissing"] = raw["Age"].isna().astype("int8")
        age_source = raw.copy()
        age_source["Title"] = data["Title"]
        data["AgeFilled"] = age_source.apply(self._impute_age_row, axis=1)
        data["AgeBand"] = pd.cut(
            data["AgeFilled"],
            bins=[-np.inf, 12, 18, 35, 50, 65, np.inf],
            labels=["Child", "Teen", "Young", "Middle", "Senior", "Elder"],
            right=True,
        ).astype(str)

        data["SibSp"] = raw["SibSp"].astype("int16")
        data["Parch"] = raw["Parch"].astype("int16")
        data["FamilySize"] = (raw["SibSp"] + raw["Parch"] + 1).astype("int16")
        data["FamilyGroup"] = _family_group(data["FamilySize"])
        data["IsAlone"] = (data["FamilySize"] == 1).astype("int8")

        data["FareMissing"] = raw["Fare"].isna().astype("int8")
        data["FareFilled"] = raw.apply(self._impute_fare_row, axis=1)
        data["FareLog"] = np.log1p(data["FareFilled"])
        data["FarePerPerson"] = data["FareFilled"] / data["FamilySize"]
        data["FareBand"] = pd.cut(
            data["FareFilled"],
            bins=self.fare_band_edges_,
            labels=self.fare_band_labels_,
            include_lowest=True,
        ).astype(str)

        data["EmbarkedFilled"] = raw["Embarked"].fillna(self.embarked_mode_).astype(str)
        data["CabinMissing"] = raw["Cabin"].isna().astype("int8")
        cabin_filled = raw["Cabin"].fillna("U").astype(str)
        data["CabinDeck"] = cabin_filled.str[0].str.upper()
        data["HasCabin"] = raw["Cabin"].notna().astype("int8")

        ticket_prefix = _extract_ticket_prefix(raw["Ticket"])
        data["TicketPrefix"] = ticket_prefix.where(
            ticket_prefix.isin(self.frequent_ticket_prefixes_),
            "OTHER",
        )
        data["SexPclass"] = data["Sex"] + "_P" + data["Pclass"].astype(str)
        data["AgePclass"] = data["AgeFilled"] * data["Pclass"]
        data["NameLength"] = raw["Name"].astype(str).str.len().astype("int16")
        data["WomenChildPriority"] = (
            (data["Sex"] == "female") | (data["AgeFilled"] < 16)
        ).astype("int8")

        expected = [
            "PassengerId",
            "Pclass",
            "Sex",
            "SexEncoded",
            "AgeFilled",
            "AgeBand",
            "SibSp",
            "Parch",
            "FamilySize",
            "FamilyGroup",
            "IsAlone",
            "FareFilled",
            "FareLog",
            "FarePerPerson",
            "FareBand",
            "EmbarkedFilled",
            "Title",
            "CabinDeck",
            "HasCabin",
            "TicketPrefix",
            "SexPclass",
            "AgePclass",
            "NameLength",
            "WomenChildPriority",
            "AgeMissing",
            "FareMissing",
            "CabinMissing",
        ]
        result = data[expected].copy()
        if result.isna().any().any():
            missing = result.isna().sum()
            raise AssertionError(
                "特征工程后仍有缺失值："
                f"{missing[missing > 0].to_dict()}"
            )
        return result

    def get_feature_names_out(
        self,
        input_features: Iterable[str] | None = None,
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(
            [
                "PassengerId",
                "Pclass",
                "Sex",
                "SexEncoded",
                "AgeFilled",
                "AgeBand",
                "SibSp",
                "Parch",
                "FamilySize",
                "FamilyGroup",
                "IsAlone",
                "FareFilled",
                "FareLog",
                "FarePerPerson",
                "FareBand",
                "EmbarkedFilled",
                "Title",
                "CabinDeck",
                "HasCabin",
                "TicketPrefix",
                "SexPclass",
                "AgePclass",
                "NameLength",
                "WomenChildPriority",
                "AgeMissing",
                "FareMissing",
                "CabinMissing",
            ],
            dtype=object,
        )


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "scaled",
                StandardScaler(),
                SCALED_NUMERIC_COLUMNS,
            ),
            (
                "numeric",
                "passthrough",
                PASSTHROUGH_NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_transform_pipeline(k: int | str = "all") -> Pipeline:
    return Pipeline(
        steps=[
            ("features", TitanicFeatureEngineer()),
            ("preprocess", make_preprocessor()),
            (
                "select",
                SelectKBest(score_func=_mutual_info_score, k=k),
            ),
        ]
    )


def make_model_pipeline(
    estimator: Any | None = None,
    *,
    k: int | str = "all",
) -> Pipeline:
    if estimator is None:
        estimator = LogisticRegression(
            max_iter=3000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    return Pipeline(
        steps=[
            ("features", TitanicFeatureEngineer()),
            ("preprocess", make_preprocessor()),
            (
                "select",
                SelectKBest(score_func=_mutual_info_score, k=k),
            ),
            ("model", estimator),
        ]
    )


def _candidate_k_values() -> list[int | str]:
    return [8, 12, 16, 20, 24, 28, 32, 36, 40, "all"]


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, list[int]]:
    outer = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    records: list[dict[str, Any]] = []
    out_of_fold_predictions = np.full(len(y), -1, dtype=int)

    for fold, (train_index, valid_index) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=RANDOM_STATE + fold,
        )
        search = GridSearchCV(
            estimator=make_model_pipeline(),
            param_grid={"select__k": _candidate_k_values()},
            scoring="accuracy",
            cv=inner,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X.iloc[train_index], y.iloc[train_index])
        predictions = search.best_estimator_.predict(X.iloc[valid_index])
        out_of_fold_predictions[valid_index] = predictions
        records.append(
            {
                "outer_fold": fold,
                "train_rows": len(train_index),
                "validation_rows": len(valid_index),
                "best_k": str(search.best_params_["select__k"]),
                "inner_best_accuracy": float(search.best_score_),
                "outer_accuracy": float(
                    accuracy_score(y.iloc[valid_index], predictions)
                ),
            }
        )

    if (out_of_fold_predictions < 0).any():
        raise AssertionError("嵌套交叉验证未覆盖所有训练样本")
    return pd.DataFrame(records), out_of_fold_predictions.tolist()


def fit_final_search(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[GridSearchCV, pd.DataFrame]:
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        estimator=make_model_pipeline(),
        param_grid={"select__k": _candidate_k_values()},
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y)
    raw_results = pd.DataFrame(search.cv_results_)
    results = pd.DataFrame(
        {
            "k": raw_results["param_select__k"].astype(str),
            "cv_accuracy_mean": raw_results["mean_test_score"].astype(float),
            "cv_accuracy_std": raw_results["std_test_score"].astype(float),
            "train_accuracy_mean": raw_results["mean_train_score"].astype(float),
            "rank": raw_results["rank_test_score"].astype(int),
        }
    ).sort_values(["rank", "k"]).reset_index(drop=True)
    results["selected"] = results["rank"].eq(1)
    return search, results


def selected_feature_details(
    fitted_pipeline: Pipeline,
) -> pd.DataFrame:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    selector = fitted_pipeline.named_steps["select"]
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=object)
    scores = np.asarray(selector.scores_, dtype=float)
    support = np.asarray(selector.get_support(), dtype=bool)
    result = pd.DataFrame(
        {
            "feature": names,
            "mutual_information": scores,
            "selected": support,
        }
    )
    return result.sort_values(
        ["selected", "mutual_information", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def transform_model_ready(
    fitted_pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    engineered = fitted_pipeline.named_steps["features"].transform(X)
    preprocessed = fitted_pipeline.named_steps["preprocess"].transform(engineered)
    selected = fitted_pipeline.named_steps["select"].transform(preprocessed)
    all_names = np.asarray(
        fitted_pipeline.named_steps["preprocess"].get_feature_names_out(),
        dtype=object,
    )
    selected_names = all_names[
        fitted_pipeline.named_steps["select"].get_support()
    ].astype(str).tolist()
    result = pd.DataFrame(selected, columns=selected_names, index=X.index)
    return result, selected_names


def _json_parameters(
    engineer: TitanicFeatureEngineer,
    train_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    return {
        "source_files": {
            "train_name": train_path.name,
            "test_name": test_path.name,
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
        },
        "age_imputation": {
            "title_pclass_sex_medians": {
                f"{title}|{pclass}|{sex}": float(value)
                for (title, pclass, sex), value in engineer.age_group_medians_.items()
            },
            "title_pclass_medians": {
                f"{title}|{pclass}": float(value)
                for (title, pclass), value in engineer.age_title_pclass_medians_.items()
            },
            "pclass_medians": {
                str(key): float(value)
                for key, value in engineer.age_pclass_medians_.items()
            },
            "overall_median": engineer.age_overall_median_,
        },
        "fare_imputation": {
            "pclass_medians": {
                str(key): float(value)
                for key, value in engineer.fare_pclass_medians_.items()
            },
            "overall_median": engineer.fare_overall_median_,
        },
        "embarked_mode": engineer.embarked_mode_,
        "ticket_prefix_min_count": engineer.ticket_prefix_min_count,
        "frequent_ticket_prefixes": engineer.frequent_ticket_prefixes_,
        "fare_band_edges": [
            "-inf" if np.isneginf(value)
            else "inf" if np.isposinf(value)
            else float(value)
            for value in engineer.fare_band_edges_
        ],
        "fare_band_labels": engineer.fare_band_labels_,
        "cv_rule": (
            "正式 CV 必须从原始数据调用 make_model_pipeline；"
            "所有学习型预处理和特征选择均在训练折内拟合"
        ),
    }


def _missing_report(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
) -> pd.DataFrame:
    before = pd.DataFrame(
        {
            "train_raw_missing": train_raw.isna().sum(),
            "test_raw_missing": test_raw.isna().sum(),
        }
    ).fillna(0)
    after = pd.DataFrame(
        {
            "train_clean_missing": train_clean.isna().sum(),
            "test_clean_missing": test_clean.isna().sum(),
        }
    ).fillna(0)
    report = before.join(after, how="outer").fillna(0).astype(int)
    report.index.name = "field"
    return report.reset_index()


def _field_dictionary() -> pd.DataFrame:
    rows = [
        ("PassengerId", "整数", "原始", "乘客唯一标识", "仅追踪和提交，不作模型特征"),
        ("Survived", "0/1", "原始（仅训练）", "是否幸存", "目标变量"),
        ("Pclass", "类别", "原始", "舱位等级", "One-Hot"),
        ("Sex", "类别", "原始", "性别", "One-Hot"),
        ("SexEncoded", "0/1", "Sex", "male=0，female=1", "候选特征"),
        ("AgeFilled", "连续", "Age", "折内分组中位数填补年龄", "折内标准化"),
        ("AgeBand", "类别", "AgeFilled", "固定年龄段", "One-Hot"),
        ("SibSp", "整数", "原始", "兄弟姐妹/配偶数", "候选特征"),
        ("Parch", "整数", "原始", "父母/子女数", "候选特征"),
        ("FamilySize", "整数", "SibSp+Parch+1", "同行家庭规模", "候选特征"),
        ("FamilyGroup", "类别", "FamilySize", "Solo/Small/Medium/Large", "One-Hot"),
        ("IsAlone", "0/1", "FamilySize", "是否独行", "候选特征"),
        ("FareFilled", "连续", "Fare", "折内按 Pclass 中位数填补", "折内标准化"),
        ("FareLog", "连续", "FareFilled", "log(1+Fare)", "候选特征"),
        ("FarePerPerson", "连续", "FareFilled/FamilySize", "人均票价", "候选特征"),
        ("FareBand", "类别", "FareFilled", "折内票价四分位", "One-Hot"),
        ("EmbarkedFilled", "类别", "Embarked", "折内众数填补登船港口", "One-Hot"),
        ("Title", "类别", "Name", "姓名称谓并合并稀有称谓", "One-Hot"),
        ("CabinDeck", "类别", "Cabin", "舱号首字母；未知为 U", "One-Hot"),
        ("HasCabin", "0/1", "Cabin", "是否有舱号记录", "候选特征"),
        ("TicketPrefix", "类别", "Ticket", "船票前缀；折内低频归 OTHER", "One-Hot"),
        ("SexPclass", "类别", "Sex×Pclass", "性别与舱位交互", "One-Hot"),
        ("AgePclass", "连续", "AgeFilled×Pclass", "年龄舱位交互", "候选特征"),
        ("NameLength", "整数", "Name", "姓名字符长度", "候选特征"),
        ("WomenChildPriority", "0/1", "Sex+AgeFilled", "妇女儿童优先规则", "候选特征"),
        ("AgeMissing", "0/1", "Age", "原始年龄是否缺失", "候选特征"),
        ("FareMissing", "0/1", "Fare", "原始票价是否缺失", "候选特征"),
        ("CabinMissing", "0/1", "Cabin", "原始舱号是否缺失", "候选特征"),
    ]
    return pd.DataFrame(
        rows,
        columns=["field", "type", "source", "definition", "modeling_guidance"],
    )


def _cleaning_rules() -> pd.DataFrame:
    rows = [
        (1, "结构校验", "严格检查列名、891/418 行、PassengerId 唯一", "PASS"),
        (2, "Age", "折内按 Title+Pclass+Sex 中位数填补并逐级回退", "PASS"),
        (3, "Fare", "折内按 Pclass 中位数填补", "PASS"),
        (4, "Embarked", "折内训练数据众数填补", "PASS"),
        (5, "Cabin", "不虚构房号；提取 HasCabin/CabinDeck，未知为 U", "PASS"),
        (6, "Name/Title", "提取称谓；同义词合并；其余归 Rare", "PASS"),
        (7, "Ticket", "提取前缀；折内计数少于 10 归 OTHER", "PASS"),
        (8, "家庭特征", "FamilySize、FamilyGroup、IsAlone", "PASS"),
        (9, "编码与缩放", "折内 One-Hot；Age/Fare 折内 StandardScaler", "PASS"),
        (10, "特征选择", "折内 SelectKBest；k 由内层 CV 选择", "PASS"),
        (11, "模型评估", "5×4 嵌套分层 CV，外层分数用于无偏估计", "PASS"),
        (12, "路径", "只使用相对路径和 CLI 参数；报告不含个人本地路径", "PASS"),
    ]
    return pd.DataFrame(rows, columns=["step", "item", "rule", "status"])


def _save_feature_selection_plot(
    selection_results: pd.DataFrame,
    output_path: Path,
    total_features: int,
) -> None:
    data = selection_results.copy()
    data["effective_k"] = data["k"].map(
        lambda value: total_features if value == "all" else int(value)
    )
    selected = data.loc[data["selected"]].iloc[0]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    ax.errorbar(
        data["effective_k"],
        data["cv_accuracy_mean"],
        yerr=data["cv_accuracy_std"],
        marker="o",
        capsize=4,
        linewidth=2,
        color="#1F5F7A",
    )
    ax.scatter(
        [selected["effective_k"]],
        [selected["cv_accuracy_mean"]],
        s=120,
        color="#D97706",
        zorder=3,
        label=f"Selected k={selected['k']}",
    )
    ax.set(
        title="Leakage-safe feature selection (inner 5-fold CV)",
        xlabel="Selected feature count",
        ylabel="Accuracy",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    train_path = Path(train_path)
    test_path = Path(test_path)
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    models_dir = output_dir / "models"
    for directory in [data_dir, reports_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    validate_raw_data(train_raw, test_raw)
    X = train_raw[RAW_FEATURE_COLUMNS].copy()
    y = train_raw["Survived"].astype("int8")

    nested_results, out_of_fold_predictions = run_nested_cv(X, y)
    final_search, selection_results = fit_final_search(X, y)
    fitted_model_pipeline = final_search.best_estimator_

    fitted_transform_pipeline = make_transform_pipeline(
        k=final_search.best_params_["select__k"]
    )
    fitted_transform_pipeline.fit(X, y)

    engineer = fitted_transform_pipeline.named_steps["features"]
    train_clean_features = engineer.transform(X)
    test_clean = engineer.transform(test_raw[RAW_FEATURE_COLUMNS])
    train_clean = train_clean_features.copy()
    train_clean.insert(1, "Survived", y.to_numpy())

    train_matrix, selected_features = transform_model_ready(
        fitted_transform_pipeline,
        X,
    )
    test_matrix, selected_features_test = transform_model_ready(
        fitted_transform_pipeline,
        test_raw[RAW_FEATURE_COLUMNS],
    )
    if selected_features != selected_features_test:
        raise AssertionError("训练集与测试集选中特征不一致")

    train_ready = pd.concat(
        [
            train_raw[["PassengerId", "Survived"]].reset_index(drop=True),
            train_matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    test_ready = pd.concat(
        [
            test_raw[["PassengerId"]].reset_index(drop=True),
            test_matrix.reset_index(drop=True),
        ],
        axis=1,
    )

    for name, frame in [
        ("train_clean", train_clean),
        ("test_clean", test_clean),
        ("train_ready", train_ready),
        ("test_ready", test_ready),
    ]:
        if frame.isna().any().any():
            raise AssertionError(f"{name} 仍有缺失值")
    train_features = train_ready.columns.drop(["PassengerId", "Survived"]).tolist()
    test_features = test_ready.columns.drop("PassengerId").tolist()
    if train_features != test_features:
        raise AssertionError("model_ready 训练/测试特征名称或顺序不一致")
    if not all(pd.api.types.is_numeric_dtype(train_ready[c]) for c in train_ready):
        raise AssertionError("train_model_ready 含非数值字段")
    if not all(pd.api.types.is_numeric_dtype(test_ready[c]) for c in test_ready):
        raise AssertionError("test_model_ready 含非数值字段")

    train_clean.to_csv(data_dir / "Titanic_train_clean.csv", index=False)
    test_clean.to_csv(data_dir / "Titanic_test_clean.csv", index=False)
    train_ready.to_csv(data_dir / "Titanic_train_model_ready.csv", index=False)
    test_ready.to_csv(data_dir / "Titanic_test_model_ready.csv", index=False)

    nested_results.to_csv(reports_dir / "nested_cv_results.csv", index=False)
    selection_results.to_csv(
        reports_dir / "feature_selection_results.csv",
        index=False,
    )
    details = selected_feature_details(fitted_transform_pipeline)
    details.to_csv(reports_dir / "feature_scores.csv", index=False)
    (reports_dir / "selected_features.txt").write_text(
        "\n".join(selected_features) + "\n",
        encoding="utf-8",
    )
    missing_report = _missing_report(
        train_raw,
        test_raw,
        train_clean,
        test_clean,
    )
    missing_report.to_csv(reports_dir / "missing_values_before_after.csv", index=False)
    _field_dictionary().to_csv(reports_dir / "field_dictionary.csv", index=False)
    _cleaning_rules().to_csv(reports_dir / "cleaning_rules.csv", index=False)

    parameters = _json_parameters(engineer, train_path, test_path)
    with (reports_dir / "preprocessing_parameters.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(parameters, file, ensure_ascii=False, indent=2)
    with (reports_dir / "oof_predictions.json").open("w", encoding="utf-8") as file:
        json.dump(out_of_fold_predictions, file)

    joblib.dump(
        fitted_transform_pipeline,
        models_dir / "titanic_b_transform_selector.joblib",
    )
    joblib.dump(
        fitted_model_pipeline,
        models_dir / "titanic_b_logistic_baseline.joblib",
    )

    preprocessed_all = fitted_transform_pipeline.named_steps["preprocess"].transform(
        fitted_transform_pipeline.named_steps["features"].transform(X)
    )
    total_feature_count = int(preprocessed_all.shape[1])
    _save_feature_selection_plot(
        selection_results,
        reports_dir / "feature_selection_cv.png",
        total_feature_count,
    )

    nested_mean = float(nested_results["outer_accuracy"].mean())
    nested_std = float(nested_results["outer_accuracy"].std(ddof=0))
    quality_report = {
        "status": "PASS",
        "raw_train_shape": list(train_raw.shape),
        "raw_test_shape": list(test_raw.shape),
        "clean_train_shape": list(train_clean.shape),
        "clean_test_shape": list(test_clean.shape),
        "model_ready_train_shape": list(train_ready.shape),
        "model_ready_test_shape": list(test_ready.shape),
        "raw_missing_values": {
            "train": int(train_raw.isna().sum().sum()),
            "test": int(test_raw.isna().sum().sum()),
        },
        "clean_missing_values": {
            "train": int(train_clean.isna().sum().sum()),
            "test": int(test_clean.isna().sum().sum()),
        },
        "model_ready_missing_values": {
            "train": int(train_ready.isna().sum().sum()),
            "test": int(test_ready.isna().sum().sum()),
        },
        "candidate_encoded_feature_count": total_feature_count,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "best_k": str(final_search.best_params_["select__k"]),
        "inner_cv_best_accuracy": float(final_search.best_score_),
        "nested_outer_accuracy_mean": nested_mean,
        "nested_outer_accuracy_std": nested_std,
        "nested_outer_fold_scores": nested_results["outer_accuracy"].tolist(),
        "oof_accuracy": float(
            accuracy_score(y, np.asarray(out_of_fold_predictions))
        ),
        "formal_evaluation_rule": (
            "使用原始 train.csv + make_model_pipeline；"
            "禁止对 model_ready.csv 再做正式 CV"
        ),
        "train_test_model_features_aligned": True,
        "all_model_ready_features_numeric": True,
        "no_absolute_personal_paths_in_deliverables": True,
        "random_state": RANDOM_STATE,
    }
    with (reports_dir / "quality_report.json").open("w", encoding="utf-8") as file:
        json.dump(quality_report, file, ensure_ascii=False, indent=2)

    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "train_clean": train_clean,
        "test_clean": test_clean,
        "train_ready": train_ready,
        "test_ready": test_ready,
        "nested_results": nested_results,
        "selection_results": selection_results,
        "feature_details": details,
        "quality_report": quality_report,
        "parameters": parameters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="原始 train.csv 路径")
    parser.add_argument("--test", required=True, help="原始 test.csv 路径")
    parser.add_argument("--output", required=True, help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_pipeline(args.train, args.test, args.output)
    report = artifacts["quality_report"]
    print("Titanic B 部分正式流水线执行完成")
    print(
        f"编码后候选特征：{report['candidate_encoded_feature_count']}；"
        f"选中特征：{report['selected_feature_count']}"
    )
    print(
        "无泄漏嵌套 CV Accuracy："
        f"{report['nested_outer_accuracy_mean']:.4f} "
        f"± {report['nested_outer_accuracy_std']:.4f}"
    )
    print(f"输出目录：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
