"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, and preprocessing steps for the
Indian bankruptcy prediction project.

Pipeline:
  1. Load the raw Excel dataset
  2. Drop multicollinear features (|r| > 0.5 threshold)
  3. Split into train/test sets (70/30, stratified)
  4. StandardScaler-normalise features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Bankrupt"
RANDOM_STATE = 42
TEST_SIZE = 0.30
CORRELATION_THRESHOLD = 0.50   # features correlated above this are dropped


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(filepath: str, sheet_name: str = "Final Data sheet ") -> pd.DataFrame:
    """
    Load the raw financial dataset from an Excel workbook.

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    sheet_name : str
        Name of the sheet containing the processed data.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with Company Name as index.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=3, index_col=0)
    # Drop the helper 'Ratios' multi-header row that Excel sometimes adds
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    print(f"Loaded dataset: {df.shape[0]} firms × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Feature selection — multicollinearity removal
# ---------------------------------------------------------------------------

def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Remove features that are mutually correlated above *threshold*.

    The algorithm iterates over the upper triangle of the correlation matrix.
    For each correlated pair it removes the feature with the lower absolute
    correlation to the target variable, preserving the more informative one.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe including the target column.
    threshold : float
        Absolute correlation cut-off.
    target : str
        Name of the target column (excluded from removal candidates).

    Returns
    -------
    pd.DataFrame
        Dataframe with highly correlated features removed.
    """
    features = df.drop(columns=[target, "Year"], errors="ignore")
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    target_corr = features.corrwith(df[target]).abs()
    to_drop = set()

    for col in upper.columns:
        correlated_cols = upper.index[upper[col] > threshold].tolist()
        for correlated_col in correlated_cols:
            # Drop the one with lower correlation to the target
            if target_corr.get(col, 0) < target_corr.get(correlated_col, 0):
                to_drop.add(col)
            else:
                to_drop.add(correlated_col)

    df_reduced = df.drop(columns=list(to_drop))
    print(
        f"Dropped {len(to_drop)} correlated features → "
        f"{df_reduced.shape[1] - 2} predictors remaining"  # -2 for target + Year
    )
    return df_reduced


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    apply_feature_selection: bool = True,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Full preprocessing pipeline: feature selection → split → scale.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (output of load_dataset).
    apply_feature_selection : bool
        If True, drop multicollinear features first.
        Set to False to use all 63 features (for ensemble models).
    test_size : float
        Proportion of data held out for testing.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
        Scaled feature matrices and binary target vectors.
    scaler : StandardScaler
        Fitted scaler (save for inference on new data).
    feature_names : list[str]
        Names of the selected features.
    """
    # Drop the Year column — not a predictive feature
    df = df.drop(columns=["Year"], errors="ignore")

    if apply_feature_selection:
        df = remove_correlated_features(df)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=feature_names,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=feature_names,
    )

    print(
        f"Train: {X_train_scaled.shape[0]} samples | "
        f"Test: {X_test_scaled.shape[0]} samples | "
        f"Class balance (train): {y_train.value_counts(normalize=True).to_dict()}"
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
