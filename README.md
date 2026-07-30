# 🚢 Titanic - Machine Learning from Disaster

> A complete, production-ready ML pipeline for the **Kaggle Titanic Competition**.
> Target accuracy: **>0.90** on the public leaderboard.

---

## 📁 Project Structure

```
Titanic-Project/
├── data/                          # Dataset files
│   ├── train.csv                  # Training set with labels
│   ├── test.csv                   # Test set for prediction
│   └── gender_submission.csv      # Sample submission format
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda_and_feature_engineering.ipynb  # EDA & feature engineering
│   └── 02_modeling_and_submission.ipynb      # Model training & submission
├── src/                           # Source modules
│   ├── __init__.py                # Package init
│   ├── data_loader.py             # Data loading (pandas/polars)
│   ├── preprocessor.py            # Missing values & encoding
│   ├── feature_engineering.py     # Feature creation
│   ├── models.py                  # Classifier definitions
│   ├── train.py                   # Training pipeline
│   └── predict.py                 # Prediction & submission
├── configs/
│   └── config.yaml                # Configuration file
├── outputs/                       # Generated outputs
│   ├── submission.csv             # Kaggle submission file
│   └── model.pkl                  # Trained model artifact
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── TASKS.md                       # Team task assignments
└── .gitignore                     # Git ignore rules
```

## 🛠️ Setup Instructions

### Prerequisites
- Python >= 3.9
- Git
- Kaggle account (for dataset download)

### Installation

```bash
# Clone the repository
git clone https://github.com/Scpkeeper/Titanic-Project.git
cd Titanic-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API)
python -c "from src.data_loader import download_kaggle_dataset; download_kaggle_dataset()"
```

### Alternative: Manual Data Download
1. Go to [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
2. Download `train.csv`, `test.csv`, `gender_submission.csv`
3. Place them in the `data/` directory

---

## 🚀 Usage Guide

### CLI Usage

```bash
# Train models with default config
python src/train.py --config configs/config.yaml

# Generate predictions
python src/predict.py --config configs/config.yaml --model outputs/model.pkl
```

### Notebook Workflow

1. Open **`notebooks/01_eda_and_feature_engineering.ipynb`**:
   - Explore data distributions
   - Analyze survival patterns
   - Walk through feature engineering steps

2. Open **`notebooks/02_modeling_and_submission.ipynb`**:
   - Train multiple classifiers
   - Compare cross-validation results
   - Generate final submission

### Configuration

Edit `configs/config.yaml` to customize:

```yaml
random_state: 42
test_size: 0.2
cv_folds: 5
model_name: "random_forest"  # or "xgboost", "lightgbm", etc.
features:
  family_size: true
  title: true
  fare_bins: true
  age_bands: true
  deck: true
  interactions: false
```

---

## 📊 Models Included

| Model | Description | Hyperparameter Tuning |
|-------|-------------|----------------------|
| Logistic Regression | Baseline linear model | GridSearchCV (C, penalty, solver) |
| Random Forest | Ensemble of decision trees | GridSearchCV (n_estimators, max_depth, ...) |
| XGBoost | Gradient boosting | GridSearchCV (learning_rate, max_depth, ...) |
| LightGBM | Light gradient boosting | GridSearchCV (num_leaves, learning_rate, ...) |
| SVM | Support Vector Machine | GridSearchCV (C, kernel, gamma) |
| MLP | Neural Network (2-layer) | GridSearchCV (hidden_layer_sizes, alpha) |

## 📈 Metrics Logged

- **Accuracy** — Overall correctness
- **Precision** — True positive rate
- **Recall** — Sensitivity
- **F1 Score** — Harmonic mean of precision/recall
- **ROC-AUC** — Area under ROC curve

All metrics computed via **Stratified K-Fold Cross-Validation** (n_splits=5).

---

## 👥 Team Collaboration

See [TASKS.md](TASKS.md) for role assignments:

| Role | Responsibilities |
|------|------------------|
| **Data Cleaning** | Missing value imputation, outlier handling |
| **Feature Engineering** | Title extraction, family features, bins, decks |
| **Model Tuning** | Hyperparameter optimization, Optuna integration |
| **Ensemble** | Stacking/blending multiple models |
| **Submission** | Final predictions, format validation, upload |

---

## 🔧 Key Features

- ✅ **Dual backend support**: pandas + polars for data loading
- ✅ **Group-based imputation**: Age filled by (Title, Sex, Pclass) median
- ✅ **Rich feature engineering**: FamilySize, IsAlone, Title, FareBin, AgeBand, Deck, Interactions
- ✅ **6 classifiers** with hyperparameter grids
- ✅ **Stratified K-Fold CV** with comprehensive metrics
- ✅ **Config-driven**: All parameters in YAML config
- ✅ **Type hints & docstrings** throughout
- ✅ **Logging module** for all src modules
- ✅ **CLI + Notebook** dual workflow support

---

## 📝 License

This project is open source under the MIT License.

## 🙏 Acknowledgments

- [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Scikit-learn, XGBoost, LightGBM communities
