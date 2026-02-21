"""
OpenFinGuard — Data Pipeline
============================
WHY THIS FILE EXISTS:
  Raw data is never model-ready. This module handles the full journey from
  raw Kaggle CSV → clean, validated, ML-ready DataFrame. Separating this
  from modeling means: (a) reproducibility, (b) testability, (c) the API
  can use the same transformations at inference time — no training/serving skew.

Dataset: Kaggle "Give Me Some Credit"
  - 150,000 borrowers, 10 features, 1 target (SeriousDlqin2yrs)
  - Real-world messiness: ~20% missing values, 1:14 class imbalance
  - Industry benchmark for credit risk ML
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Column Definitions ─────────────────────────────────────────────────────
# WHY: Explicit column naming prevents silent errors if Kaggle changes CSV format
TARGET = "SeriousDlqin2yrs"

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",  # credit utilization ratio
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",  # 30-59 day delinquencies
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",  # 60-89 day delinquencies
    "NumberOfDependents",
]

# ── Feature Groups (used later for fairness & explainability) ──────────────
BEHAVIORAL_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

DEMOGRAPHIC_FEATURES = ["age", "NumberOfDependents"]

FINANCIAL_FEATURES = [
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
]


def download_dataset() -> Path:
    """
    Download dataset via Kaggle API.
    WHY KAGGLE API: Programmatic download = reproducible pipeline.
    Anyone cloning the repo can reproduce the exact dataset with one command.

    Requires: KAGGLE_USERNAME and KAGGLE_KEY in .env
    Run: kaggle competitions download -c GiveMeSomeCredit
    """
    output_path = RAW_DIR / "cs-training.csv"
    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}, skipping download.")
        return output_path

    logger.info("Downloading dataset from Kaggle...")
    os.system(
        "kaggle competitions download -c GiveMeSomeCredit -p data/raw/ --unzip"
    )

    if not output_path.exists():
        raise FileNotFoundError(
            "Download failed. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set in .env\n"
            "Or manually download from: https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            "Place cs-training.csv in data/raw/"
        )

    logger.success(f"Dataset downloaded to {output_path}")
    return output_path


def load_raw(path: Path = None) -> pd.DataFrame:
    """Load raw CSV and perform initial sanity checks."""
    path = path or RAW_DIR / "cs-training.csv"
    df = pd.read_csv(path, index_col=0)  # first col is row index in this dataset
    logger.info(f"Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Target distribution:\n{df[TARGET].value_counts(normalize=True).round(3)}")
    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """
    WHY: Before cleaning, document what's broken.
    This audit becomes part of your research report — it shows you understood
    the data problems rather than blindly imputing them.
    """
    audit = {
        "shape": df.shape,
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "target_imbalance_ratio": (
            df[TARGET].value_counts()[0] / df[TARGET].value_counts()[1]
        ),
        "outlier_summary": {},
    }

    # Detect outliers using IQR method
    for col in FEATURE_COLS:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)).sum()
            audit["outlier_summary"][col] = int(outliers)

    logger.info(f"Data Quality Audit:\n"
                f"  Missing values: MonthlyIncome={audit['missing_pct'].get('MonthlyIncome', 0):.1f}%, "
                f"NumberOfDependents={audit['missing_pct'].get('NumberOfDependents', 0):.1f}%\n"
                f"  Class imbalance ratio: {audit['target_imbalance_ratio']:.1f}:1\n"
                f"  Duplicates: {audit['duplicates']}")
    return audit


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY EACH STEP:
    1. Remove duplicates — duplicate rows bias model toward repeated patterns
    2. Cap outliers — extreme values (e.g., age=0, utilization=9999) are data errors
    3. Impute missing values — median for income (skewed dist), mode for dependents
    4. Domain validation — credit rules that any domain expert would enforce
    """
    df = df.copy()
    initial_rows = len(df)

    # Step 1: Remove exact duplicates
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

    # Step 2: Domain-based outlier capping
    # WHY: Age of 0 or 120 is a data entry error, not a real borrower
    df["age"] = df["age"].clip(lower=18, upper=100)

    # WHY: Utilization > 1 is theoretically possible (overlimit) but >13 is noise
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(lower=0, upper=13)

    # WHY: DebtRatio > 1 means spending more than earning — valid but cap extreme noise
    df["DebtRatio"] = df["DebtRatio"].clip(lower=0, upper=50)

    # WHY: Delinquency counts > 20 in 2 years = data error (only ~24 months exist)
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]:
        df[col] = df[col].clip(lower=0, upper=20)

    # Step 3: Missing value imputation
    # WHY MEDIAN for income: Income is right-skewed — median is more representative than mean
    monthly_income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(monthly_income_median)

    # WHY MODE for dependents: It's a count variable — median/mean gives non-integer
    dependents_mode = df["NumberOfDependents"].mode()[0]
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dependents_mode)

    logger.success(f"Data cleaned: {len(df):,} rows remaining")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY FEATURE ENGINEERING:
    Raw features are what happened. Engineered features capture WHY it happened.
    These derived signals often have higher predictive power than raw inputs.
    """
    df = df.copy()

    # Total delinquency burden — aggregate payment behavior signal
    # WHY: A borrower with 3 types of delinquency is riskier than one with 1 type
    df["TotalDelinquencies"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
    )

    # Delinquency severity ratio — captures how bad (not just how frequent)
    # WHY: 90-day late is far worse than 30-day; this weights severity
    df["DelinquencySeverityScore"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1
        + df["NumberOfTime60-89DaysPastDueNotWorse"] * 2
        + df["NumberOfTimes90DaysLate"] * 3
    )

    # Monthly debt payment estimate
    # WHY: DebtRatio × Income gives absolute debt burden, not just ratio
    df["EstimatedMonthlyDebt"] = df["DebtRatio"] * df["MonthlyIncome"]

    # Disposable income proxy
    # WHY: A borrower earning $10k with $9k debt is riskier than $5k income $1k debt
    df["DisposableIncome"] = df["MonthlyIncome"] - df["EstimatedMonthlyDebt"]

    # Credit line density — normalized credit exposure
    # WHY: 10 credit lines at age 25 is very different from 10 lines at age 65
    df["CreditLineDensity"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["age"] + 1)

    # Age group — used for fairness analysis (not as a model feature directly)
    # WHY: Fairness analysis requires demographic buckets to measure disparate impact
    df["AgeGroup"] = pd.cut(
        df["age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "60+"],
    )

    logger.info(f"Engineered 5 new features. Total features: {len(df.columns)}")
    return df


def prepare_ml_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    apply_smote: bool = True,
    random_state: int = 42,
) -> dict:
    """
    WHY THIS SPLIT STRATEGY:
    - Train/Val/Test (70/10/20): Val for hyperparameter tuning, Test for final evaluation
    - Never tune on test set — that's data leakage
    - SMOTE only on TRAIN set — applying it to val/test would inflate metrics artificially

    WHY SMOTE over class_weight:
    - SMOTE creates synthetic minority samples → model sees more default patterns
    - class_weight just reweights loss — less effective for tree models
    - We'll compare both in the modeling stage
    """
    # Select final feature set
    ml_features = FEATURE_COLS + [
        "TotalDelinquencies",
        "DelinquencySeverityScore",
        "EstimatedMonthlyDebt",
        "DisposableIncome",
        "CreditLineDensity",
    ]

    # Keep AgeGroup separate for fairness analysis
    fairness_col = "AgeGroup"

    X = df[ml_features]
    y = df[TARGET]
    age_groups = df[fairness_col]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test, ag_trainval, ag_test = train_test_split(
        X, y, age_groups, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, ag_train, ag_val = train_test_split(
        X_trainval,
        y_trainval,
        ag_trainval,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_trainval,
    )

    logger.info(
        f"Split sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )
    logger.info(
        f"Train class balance before SMOTE: {y_train.value_counts().to_dict()}"
    )

    # Apply SMOTE to training set only
    if apply_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"After SMOTE — Train: {len(X_train_res):,} | "
            f"Class balance: {pd.Series(y_train_res).value_counts().to_dict()}"
        )
    else:
        X_train_res, y_train_res = X_train, y_train

    # Scale features (needed for Logistic Regression baseline)
    # WHY: Tree models don't need scaling, but LR does. We store both.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        # Raw (for tree models)
        "X_train": X_train_res,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train_res,
        "y_val": y_val,
        "y_test": y_test,
        # Scaled (for linear models)
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        # Fairness metadata
        "age_groups_val": ag_val,
        "age_groups_test": ag_test,
        # Artifacts
        "scaler": scaler,
        "feature_names": ml_features,
    }


def _make_json_serializable(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    WHY: pandas/numpy operations return numpy scalars (int64, float64) which
    Python's json module cannot serialize — this converter handles all cases.
    """
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [_make_json_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_processed(datasets: dict, audit: dict) -> None:
    """Save processed datasets and audit report for reproducibility."""
    import joblib
    import json

    # Save DataFrames
    datasets["X_train"].to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    datasets["X_val"].to_csv(PROCESSED_DIR / "X_val.csv", index=False)
    datasets["X_test"].to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    datasets["y_train"].to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    datasets["y_val"].to_csv(PROCESSED_DIR / "y_val.csv", index=False)
    datasets["y_test"].to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    # Save scaler for inference-time use
    joblib.dump(datasets["scaler"], PROCESSED_DIR / "scaler.joblib")

    # Save audit — convert all numpy types before serializing
    audit_serializable = _make_json_serializable(
        {k: v for k, v in audit.items() if k != "shape"}
    )
    with open(PROCESSED_DIR / "data_audit.json", "w") as f:
        json.dump(audit_serializable, f, indent=2)

    logger.success(f"All processed data saved to {PROCESSED_DIR}")


def run_pipeline(raw_path: Path = None) -> dict:
    """
    Full pipeline entry point.
    WHY A SINGLE ENTRY POINT: Makes it callable from notebooks, CLI, and tests.
    """
    logger.info("=" * 60)
    logger.info("OpenFinGuard — Data Pipeline Starting")
    logger.info("=" * 60)

    df_raw = load_raw(raw_path)
    audit = audit_data_quality(df_raw)
    df_clean = clean_data(df_raw)
    df_features = engineer_features(df_clean)
    datasets = prepare_ml_dataset(df_features)
    save_processed(datasets, audit)

    logger.success("Pipeline complete. Ready for modeling.")
    return datasets


if __name__ == "__main__":
    run_pipeline()