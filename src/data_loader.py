"""
Data Loader Module
==================
Download and load Titanic dataset (train.csv, test.csv).
Supports both pandas and polars backends.
"""

import logging
import os
from pathlib import Path
from typing import Union, Optional, Literal

logger = logging.getLogger(__name__)

# Try optional backends
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


BackendType = Literal["pandas", "polars"]
DataFrameType = Union["pd.DataFrame", "pl.DataFrame"]


def load_data(
    data_dir: Union[str, Path],
    filename: str,
    backend: BackendType = "pandas",
) -> DataFrameType:
    """
    Load a CSV file from the data directory.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the CSV file.
    filename : str
        Name of the CSV file (e.g., 'train.csv').
    backend : {'pandas', 'polars'}
        Which library to use for loading.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ImportError
        If the requested backend is not installed.
    """
    filepath = Path(data_dir) / filename

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    if backend == "polars":
        if not HAS_POLARS:
            raise ImportError("Polars is not installed. Run: pip install polars")
        logger.info(f"Loading {filename} with Polars backend")
        return pl.read_csv(filepath)
    else:
        if not HAS_PANDAS:
            raise ImportError("Pandas is not installed. Run: pip install pandas")
        logger.info(f"Loading {filename} with Pandas backend")
        return pd.read_csv(filepath)


def load_train_test(
    data_dir: Union[str, Path],
    backend: BackendType = "pandas",
) -> tuple[DataFrameType, DataFrameType]:
    """
    Load both train and test datasets.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the CSV files.
    backend : {'pandas', 'polars'}
        Which library to use for loading.

    Returns
    -------
    tuple of (train_df, test_df)
    """
    train_df = load_data(data_dir, "train.csv", backend=backend)
    test_df = load_data(data_dir, "test.csv", backend=backend)
    return train_df, test_df


def download_kaggle_dataset(
    competition: str = "titanic",
    output_dir: Union[str, Path] = "./data",
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
) -> Path:
    """
    Download Titanic dataset from Kaggle using the Kaggle API.

    Requires kaggle CLI and valid credentials (~/.kaggle/kaggle.json).

    Parameters
    ----------
    competition : str
        Kaggle competition name.
    output_dir : str or Path
        Directory to save downloaded files.
    kaggle_username : str, optional
        Kaggle username (overrides env/config).
    kaggle_key : str, optional
        Kaggle API key (overrides env/config).

    Returns
    -------
    Path
        Output directory path.
    """
    import subprocess

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set environment variables if provided
    env = os.environ.copy()
    if kaggle_username:
        env["KAGGLE_USERNAME"] = kaggle_username
    if kaggle_key:
        env["KAGGLE_KEY"] = kaggle_key

    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(output_path)]
    logger.info(f"Downloading Kaggle dataset: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logger.error(f"Kaggle download failed: {result.stderr}")
        raise RuntimeError(f"Kaggle API error: {result.stderr}")

    # Unzip if needed
    zip_file = output_path / f"{competition}.zip"
    if zip_file.exists():
        import zipfile
        logger.info(f"Extracting {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(output_path)
        zip_file.unlink()

    logger.info(f"Dataset downloaded to {output_path}")
    return output_path


def get_feature_columns(df: DataFrameType, exclude_cols: list[str] = None) -> list[str]:
    """
    Return feature column names (excluding ID and target columns).

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input dataframe.
    exclude_cols : list of str, optional
        Columns to exclude. Defaults to ['PassengerId', 'Survived', 'Name',
        'Ticket'].

    Returns
    -------
    list of str
        Feature column names.
    """
    if exclude_cols is None:
        exclude_cols = ["PassengerId", "Survived", "Name", "Ticket"]

    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        return [c for c in df.columns if c not in exclude_cols]
    elif HAS_POLARS and isinstance(df, pl.DataFrame):
        return [c for c in df.columns if c not in exclude_cols]
    else:
        raise TypeError(f"Unsupported dataframe type: {type(df)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    if (data_dir / "train.csv").exists():
        train, test = load_train_test(data_dir)
        print(f"Train shape: {train.shape}")
        print(f"Test shape:  {test.shape}")
    else:
        print("Data files not found. Run download_kaggle_dataset() first.")
