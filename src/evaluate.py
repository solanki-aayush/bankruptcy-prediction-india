"""
evaluate.py
-----------
Computes and visualises model evaluation metrics.

Metrics used in the paper:
  • Accuracy
  • ROC-AUC
  • F1-score (macro, to account for class imbalance)
  • Cross-validation accuracy (in train.py)
  • Confusion matrix
  • ROC curve
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Consistent colour palette
PALETTE = {
    "primary": "#1f4e79",
    "secondary": "#2e75b6",
    "accent": "#ed7d31",
    "positive": "#70ad47",
    "negative": "#ff0000",
}


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute accuracy, ROC-AUC, and macro F1-score for an sklearn model.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict with keys: accuracy, roc_auc, f1_score
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1_score": f1_score(y_test, y_pred, average="macro"),
    }


def compute_metrics_numpy(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Same as compute_metrics but accepts raw numpy arrays (for LSTM output).

    Parameters
    ----------
    y_true : np.ndarray — binary ground truth
    y_pred_proba : np.ndarray — predicted probabilities for class 1
    """
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "f1_score": f1_score(y_true, y_pred, average="macro"),
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save: bool = True,
) -> None:
    """Plot and optionally save a styled confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Non-Bankrupt", "Bankrupt"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        fname = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str = "ROC Curves",
    filename: str = "roc_curves.png",
    save: bool = True,
) -> None:
    """
    Plot ROC curves for multiple models on a single axes.

    Parameters
    ----------
    models : dict
        {name: fitted sklearn estimator}.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = list(PALETTE.values())

    for idx, (name, model) in enumerate(models.items()):
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})",
                color=colours[idx % len(colours)], lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "Random Forest",
    save: bool = True,
) -> None:
    """
    Horizontal bar chart of the top-N feature importances.

    Only applicable to tree-based models with .feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"  {model_name} does not expose feature_importances_; skipping.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind="barh", ax=ax, color=PALETTE["secondary"])
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()

    if save:
        fname = model_name.lower().replace(" ", "_") + "_feature_importance.png"
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(pca_summary: pd.DataFrame, save: bool = True) -> None:
    """Bar chart of explained variance per principal component."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        pca_summary["component"],
        pca_summary["explained_variance_ratio"],
        color=[PALETTE["secondary"] if v < 0.12 else PALETTE["primary"]
               for v in pca_summary["explained_variance_ratio"]],
    )
    ax.plot(
        pca_summary["component"],
        pca_summary["cumulative_variance"],
        "o-", color=PALETTE["accent"], label="Cumulative variance",
    )
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Ratio", fontsize=12)
    ax.set_title("Proportion of Variance Explained by Each Principal Component",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURES_DIR, "pca_variance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_classification_report(model, X_test, y_test, model_name: str) -> None:
    """Print sklearn classification report."""
    y_pred = model.predict(X_test)
    print(f"\n{'='*55}")
    print(f"Classification Report — {model_name}")
    print('='*55)
    print(classification_report(y_test, y_pred, target_names=["Non-Bankrupt", "Bankrupt"]))


def summarise_results(results_list: list[dict]) -> pd.DataFrame:
    """
    Combine multiple result dicts into a tidy comparison DataFrame.

    Parameters
    ----------
    results_list : list[dict]
        Each dict must contain keys: model, accuracy, roc_auc, f1_score.
        Optional key: cross_val_mean.

    Returns
    -------
    pd.DataFrame sorted by accuracy descending.
    """
    df = pd.DataFrame(results_list)
    for col in ["accuracy", "roc_auc", "f1_score"]:
        df[col] = df[col].round(4)
    if "cross_val_mean" in df.columns:
        df["cross_val_mean"] = df["cross_val_mean"].round(4)
    return df.sort_values("accuracy", ascending=False).set_index("model")
