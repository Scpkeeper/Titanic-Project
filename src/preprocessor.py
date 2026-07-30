"""
Preprocessor Module
===================
Handle missing values, encode categorical features,
and prepare data for model training.
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class TitanicPreprocessor:
    """
    Preprocessing pipeline for the Titanic dataset.

    Handles missing value imputation, categorical encoding,
    feature scaling, and deck extraction from Cabin.
    """

    def __init__(
        self,
        fill_age_by_group: bool = True,
        encode_categorical: bool = True,
        scale_features: bool = True,
        extract_deck: bool = True,
    ):
        self.fill_age_by_group = fill_age_by_group
        self.encode_categorical = encode_categorical
        self.scale_features = scale_features
        self.extract_deck = extract_deck

        # Fitted parameters
        self.age_medians_: dict[tuple, float] = {}
        self.fare_median_: float = 0.0
        self.embarked_mode_: str = "S"
        self.scaler_ = StandardScaler()
        self.label_encoders_: dict[str, LabelEncoder] = {}
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "TitanicPreprocessor":
        """
        Fit preprocessor on training data.

        Computes medians/modes for imputation, fits label encoders and scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe (must contain 'Survived' target column).

        Returns
        -------
        self
        """
        logger.info("Fitting preprocessor on training data...")

        # Age median by group (Title/Sex/Pclass)
        if self.fill_age_by_group and "Age" in df.columns:
            # Extract title first
            df = self._extract_title(df)
            groups = df.groupby(["Title", "Sex", "Pclass"])["Age"].median()
            self.age_medians_ = groups.to_dict()
            global_age_median = df["Age"].median()
            # Fill NaN keys with global median
            self.age_medians_ = {
                k: (v if not np.isnan(v) else global_age_median)
                for k, v in self.age_medians_.items()
            }
            logger.info(f"Computed age medians for {len(self.age_medians_)} groups")

        # Fare median
        if "Fare" in df.columns:
            self.fare_median_ = df["Fare"].median()

        # Embarked mode
        if "Embarked" in df.columns:
            mode_result = df["Embarked"].mode()
            self.embarked_mode_ = mode_result.iloc[0] if len(mode_result) > 0 else "S"
            logger.info(f"Embarked mode: {self.embarked_mode_}")

        # Fit label encoders
        if self.encode_categorical:
            for col in ["Sex", "Embarked"]:
                if col in df.columns:
                    le = LabelEncoder()
                    # Fill NaN before fitting
                    series = df[col].fillna("missing")
                    le.fit(series)
                    self.label_encoders_[col] = le
                    logger.info(f"Fitted LabelEncoder for '{col}': {list(le.classes_)}")

        # Fit scaler
        numeric_cols = self._get_numeric_cols(df)
        if self.scale_features and numeric_cols:
            temp_df = df.copy()
            temp_df = self._fill_missing(temp_df)
            self.scaler_.fit(temp_df[numeric_cols])
            logger.info(f"Fitted StandardScaler on {len(numeric_cols)} numeric columns")

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe using fitted preprocessor.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to transform.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe ready for modeling.
        """
        if not self.fitted_:
            raise ValueError("Preprocessor has not been fitted. Call fit() first.")

        result = df.copy()

        # Extract features that depend on raw data
        result = self._extract_title(result)

        if self.extract_deck and "Cabin" in result.columns:
            result = self._extract_deck(result)

        # Fill missing values
        result = self._fill_missing(result)

        # Encode categoricals
        if self.encode_categorical:
            result = self._encode_categorical(result)

        # Create derived features
        result = self._create_derived_features(result)

        # Scale numeric features
        if self.scale_features:
            result = self._scale_numeric(result)

        logger.info(f"Transformed data shape: {result.shape}")
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract title from Name column."""
        if "Name" not in df.columns:
            return df

        def get_title(name: str) -> str:
            match = re.search(r",\s*([^\.]+)\.", str(name))
            if match:
                return match.group(1).strip()
            return "Unknown"

        df["Title"] = df["Name"].apply(get_title)

        # Group rare titles
        title_map = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Royalty", "Countess": "Royalty", "Sir": "Royalty",
            "Jonkheer": "Royalty", "Don": "Royalty", "Dona": "Royalty",
            "Capt": "Officer", "Col": "Officer", "Major": "Officer",
            "Dr": "Dr", "Rev": "Rev",
        }
        df["Title"] = df["Title"].replace(title_map)
        return df

    def _extract_deck(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract deck letter from Cabin."""
        def get_deck(cabin: str) -> str:
            if pd.isna(cabin) or str(cabin).strip() == "":
                return "M"  # Missing
            return str(cabin)[0].upper()

        df["Deck"] = df["Cabin"].apply(get_deck)
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using fitted statistics."""

        # Age — impute by group median
        if "Age" in df.columns and self.age_medians_:
            global_median = (
                [v for v in self.age_medians_.values() if not np.isnan(v)][0]
                if self.age_medians_
                else 28.0
            )

            def impute_age(row):
                if pd.notna(row["Age"]):
                    return row["Age"]
                key = (row.get("Title", ""), row.get("Sex", ""), row.get("Pclass", 3))
                return self.age_medians_.get(key, global_median)

            df["Age"] = df.apply(impute_age, axis=1)

        # Embarked — mode
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)

        # Fare — median
        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(self.fare_median_)

        # Cabin — already handled via Deck extraction
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns."""
        for col, le in self.label_encoders_.items():
            if col in df.columns:
                # Handle unseen labels
                filled = df[col].fillna("missing")
                df[col + "_encoded"] = filled.apply(
                    lambda x: x if x in le.classes_ else "missing"
                )
                df[col + "_encoded"] = le.transform(df[col + "_encoded"])
                logger.info(f"Encoded '{col}' -> '{col}_encoded'")
        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create FamilySize, IsAlone, FareBin, AgeBand."""
        # FamilySize
        if "SibSp" in df.columns and "Parch" in df.columns:
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

        # IsAlone
        if "FamilySize" in df.columns:
            df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        # Fare bins
        if "Fare" in df.columns:
            df["FareBin"] = pd.qcut(
                df["Fare"], q=5, labels=False, duplicates="drop"
            ).fillna(2).astype(int)

        # Age bands
        if "Age" in df.columns:
            df["AgeBand"] = pd.cut(
                df["Age"],
                bins=[0, 12, 20, 40, 60, 100],
                labels=[0, 1, 2, 3, 4],
            ).fillna(2).astype(int)

        return df

    def _get_numeric_cols(self, df: pd.DataFrame) -> list[str]:
        """Get columns suitable for scaling."""
        skip = {"PassengerId", "Survived", "Name", "Ticket", "Cabin"}
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric if c not in skip]

    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to numeric columns."""
        numeric_cols = self._get_numeric_cols(df)
        existing = [c for c in numeric_cols if c in df.columns]
        if existing:
            df[existing] = self.scaler_.transform(df[existing])
        return df

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """
        Get final feature column names after transformation.

        Parameters
        ----------
        df : pd.DataFrame
            Reference dataframe (pre-transformation).

        Returns
        -------
        list of str
            Feature names usable for model training.
        """
        sample = self.transform(df.head())
        exclude = {"PassengerId", "Survived", "Name", "Ticket", "Cabin"}
        return [c for c in sample.columns if c not in exclude]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train.csv"

    if train_path.exists():
        df = pd.read_csv(train_path)
        prep = TitanicPreprocessor()
        processed = prep.fit_transform(df)
        print(f"Original shape:  {df.shape}")
        print(f"Processed shape: {processed.shape}")
        print(f"Features: {prep.get_feature_names(df)}")
    else:
        print("train.csv not found.")
