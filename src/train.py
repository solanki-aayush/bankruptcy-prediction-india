"""
train.py
--------
Trains all models studied in the paper:

  Statistical baseline
  --------------------
  • Logistic Regression (on 28-feature correlation-filtered data)

  Classical ML models (on full 63-feature data, with and without PCA)
  --------------------------------------------------------------------
  • K-Nearest Neighbours (KNN)
  • Random Forest
  • XGBoost (GradientBoostingClassifier)

  Deep learning
  -------------
  • LSTM (on full 63-feature data reshaped as a single time-step sequence)

All models are serialised to /models/ after training.
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_classical_models() -> dict:
    """Return a dict of {name: unfitted sklearn estimator}."""
    return {
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }


def get_logistic_model() -> LogisticRegression:
    """Logistic regression baseline (regularised to handle small n)."""
    return LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_and_evaluate_classical(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = 10,
    label: str = "",
) -> pd.DataFrame:
    """
    Fit each model, run cross-validation, and collect summary metrics.

    Parameters
    ----------
    models : dict
        {name: sklearn estimator}.
    X_train, X_test : pd.DataFrame
        Feature matrices (already scaled).
    y_train, y_test : pd.Series
        Binary target vectors.
    cv_folds : int
        Number of cross-validation folds.
    label : str
        Suffix appended to model name in the results table (e.g. "PCA").

    Returns
    -------
    pd.DataFrame
        Summary table with accuracy, cross-val score, per model.
    """
    from evaluate import compute_metrics  # local import to avoid circular dep

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
        metrics = compute_metrics(model, X_test, y_test)
        metrics["cross_val_mean"] = cv_scores.mean()
        metrics["model"] = f"{name}{(' [' + label + ']') if label else ''}"

        rows.append(metrics)

        # Persist
        tag = name.lower().replace(" ", "_") + (f"_{label.lower()}" if label else "")
        with open(os.path.join(MODELS_DIR, f"{tag}.pkl"), "wb") as f:
            pickle.dump(model, f)

        print(f"  ✓ {metrics['model']:35s} Acc={metrics['accuracy']:.4f}  AUC={metrics['roc_auc']:.4f}")

    return pd.DataFrame(rows).set_index("model")


def train_logistic(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[LogisticRegression, pd.DataFrame]:
    """
    Train and summarise the logistic regression baseline.

    Returns the fitted model and a one-row metrics DataFrame.
    """
    from evaluate import compute_metrics

    model = get_logistic_model()
    model.fit(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test)
    metrics["model"] = "Logistic Regression"

    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"  ✓ Logistic Regression  Acc={metrics['accuracy']:.4f}")
    return model, pd.DataFrame([metrics]).set_index("model")


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def train_lstm(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
):
    """
    Build, train, and evaluate a single-layer LSTM classifier.

    Input arrays are reshaped to (samples, timesteps=1, features) as the
    paper treats each firm's annual snapshot as a single time step.

    Parameters
    ----------
    X_train, X_test : np.ndarray  (2-D)
        Scaled feature matrices from StandardScaler.
    y_train, y_test : np.ndarray  (1-D)
        Binary labels.
    epochs, batch_size : int
        Keras training hyperparameters.

    Returns
    -------
    model : tf.keras.Model
        Trained LSTM model.
    history : tf.keras.callbacks.History
    metrics : dict
        accuracy, roc_auc, f1_score for test set.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError(
            "TensorFlow is required for LSTM training. "
            "Install it with: pip install tensorflow"
        )

    from evaluate import compute_metrics_numpy

    # Reshape to (samples, 1 timestep, features)
    X_train_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = Sequential([
        LSTM(units=50, input_shape=(1, X_train.shape[1])),
        Dense(units=1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_3d, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred_proba = model.predict(X_test_3d, verbose=0).flatten()
    metrics = compute_metrics_numpy(y_test, y_pred_proba)
    metrics["model"] = "LSTM"

    model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))
    print(f"  ✓ LSTM  Acc={metrics['accuracy']:.4f}  AUC={metrics['roc_auc']:.4f}")
    return model, history, metrics
