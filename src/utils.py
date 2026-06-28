"""
utils.py
--------
Shared utility functions used across the project.
"""

from __future__ import annotations

import os
import pickle
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Matplotlib / Seaborn defaults
# ---------------------------------------------------------------------------

def set_plot_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
    })


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str) -> None:
    """Pickle a fitted model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {path}")


def load_model(path: str):
    """Load a pickled model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(results: pd.DataFrame, path: str) -> None:
    """Save a results DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)
    print(f"  Results saved: {path}")


def load_results(path: str) -> pd.DataFrame:
    """Load a results CSV back as a DataFrame."""
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def class_distribution(y: pd.Series, label: str = "") -> None:
    """Print class counts and proportions."""
    counts = y.value_counts()
    proportions = y.value_counts(normalize=True).round(3)
    tag = f"[{label}] " if label else ""
    print(f"{tag}Class distribution:")
    for cls in counts.index:
        print(f"  {'Bankrupt' if cls == 1 else 'Non-Bankrupt':15s}: "
              f"{counts[cls]:4d} ({proportions[cls]*100:.1f}%)")


def feature_summary(df: pd.DataFrame, target_col: str = "Bankrupt") -> pd.DataFrame:
    """
    Return a summary DataFrame: mean by class for each feature.
    Useful for a quick sanity-check on predictive direction.
    """
    return df.groupby(target_col).mean().T


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_run(params: dict, results: dict, log_path: str = "results/run_log.json") -> None:
    """
    Append a JSON entry to a run log (for experiment tracking without MLflow).

    Parameters
    ----------
    params : dict   — hyperparameters / settings used in this run
    results : dict  — metric values from this run
    log_path : str  — path to the JSON log file
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(), "params": params, "results": results}

    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
