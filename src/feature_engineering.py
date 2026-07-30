"""
Feature Engineering Module
==========================
Create advanced features for the Titanic dataset.

Features created:
- FamilySize = SibSp + Parch + 1
- IsAlone (binary)
- Title extracted from Name
- Fare bins (qcut-based)
- Age bands (cut-based)
- Cabin deck letter
- Interaction features (optional)
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for Titanic data.

    All transformations are deterministic and stateless,
    making this safe to apply identically to train and test sets.
    """

    def __init__(
        self,
        create_family_features: bool = True,
        create_title_feature: bool = True,
        create_fare_bins: bool = True,
        create_age_bands: bool = True,
        create_deck_feature: bool = True,
        create_interactions: bool = False,
    ):
        self.create_family_features = create_family_features
        self.create_title_feature = create_title_feature
        self.create_fare_bins = create_fare_bins
        self.create_age_bands = create_age_bands
        self.create_deck_feature = create_deck_feature
        self.create_interactions = create_interactions

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all enabled feature engineering steps.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with additional engineered features.
        """
        result = df.copy()

        if self.create_title_feature:
            result = self._add_title(result)

        if self.create_family_features:
            result = self._add_family_features(result)

        if self.create_fare_bins:
            result = self._add_fare_bins(result)

        if self.create_age_bands:
            result = self._add_age_bands(result)

        if self.create_deck_feature:
            result = self._add_deck(result)

        if self.create_interactions:
            result = self._add_interactions(result)

        new_cols = set(result.columns) - set(df.columns)
        logger.info(f"Added {len(new_cols)} engineered features: {sorted(new_cols)}")
        return result

    # ------------------------------------------------------------------
    # Feature extraction functions
    # ------------------------------------------------------------------

    @staticmethod
    def extract_title(name: str) -> str:
        """Extract honorific title from a passenger name."""
        match = re.search(r",\s*([^\.]+)\.", str(name))
        if match:
            return match.group(1).strip()
        return "Unknown"

    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalize rare titles into common categories.

        Categories: Mr, Mrs, Miss, Master, Royalty, Officer, Dr, Rev
        """
        title_map = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Royalty", "Countess": "Royalty", "Sir": "Royalty",
            "Jonkheer": "Royalty", "Don": "Royalty", "Dona": "Royalty",
            "Capt": "Officer", "Col": "Officer", "Major": "Officer",
            "Dr": "Dr", "Rev": "Rev",
        }
        return title_map.get(title, title)

    @staticmethod
    def extract_deck(cabin: str) -> str:
        """Extract deck letter from cabin number."""
        if pd.isna(cabin) or str(cabin).strip() == "":
            return "M"
        return str(cabin)[0].upper()

    # ------------------------------------------------------------------
    # Internal transformers
    # ------------------------------------------------------------------

    def _add_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized Title column."""
        df["Title"] = df["Name"].apply(self.extract_title).apply(self.normalize_title)
        return df

    def _add_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add FamilySize and IsAlone features."""
        df["FamilySize"] = df.get("SibSp", 0) + df.get("Parch", 0) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        return df

    def _add_fare_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fare quantile bins."""
        if "Fare" not in df.columns:
            return df
        df["FareBin"] = pd.qcut(
            df["Fare"].fillna(df["Fare"].median()),
            q=5,
            labels=["VeryLow", "Low", "Medium", "High", "VeryHigh"],
            duplicates="drop",
        ).astype(str)
        return df

    def _add_age_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add age band categories."""
        if "Age" not in df.columns:
            return df
        df["AgeBand"] = pd.cut(
            df["Age"].fillna(df["Age"].median()),
            bins=[0, 12, 20, 40, 60, 120],
            labels=["Child", "Teen", "Adult", "MiddleAge", "Senior"],
        ).astype(str)
        return df

    def _add_deck(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Deck feature from Cabin."""
        if "Cabin" not in df.columns:
            return df
        df["Deck"] = df["Cabin"].apply(self.extract_deck)
        return df

    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables."""
        # Sex * Pclass
        if "Sex" in df.columns and "Pclass" in df.columns:
            df["Sex_Pclass"] = df["Sex"] + "_" + df["Pclass"].astype(str)

        # Title * Pclass
        if "Title" in df.columns and "Pclass" in df.columns:
            df["Title_Pclass"] = df["Title"] + "_" + df["Pclass"].astype(str)

        # Age * Pclass (continuous interaction)
        if "Age" in df.columns and "Pclass" in df.columns:
            df["Age_x_Pclass"] = df["Age"].fillna(df["Age"].median()) * df["Pclass"]

        # FamilySize * Pclass
        if "FamilySize" in df.columns and "Pclass" in df.columns:
            df["FamilySize_x_Pclass"] = df["FamilySize"] * df["Pclass"]

        return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function: apply full feature engineering with defaults.

    Parameters
    ----------
    df : pd.DataFrame
        Input Titanic dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with all default engineered features added.
    """
    engineer = FeatureEngineer(create_interactions=True)
    return engineer.transform(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path
    data_dir = Path(__file__).parent.parent / "data"
    train_path = data_dir / "train.csv"

    if train_path.exists():
        df = pd.read_csv(train_path)
        engineered = create_all_features(df)
        print(f"Original:  {df.shape}")
        print(f"Engineered: {engineered.shape}")
        print("\nNew columns:")
        for c in sorted(set(engineered.columns) - set(df.columns)):
            print(f"  + {c}")
    else:
        print("train.csv not found.")
