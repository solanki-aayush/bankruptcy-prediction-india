"""
feature_engineering.py
-----------------------
Dimensionality reduction via PCA, applied after the preprocessing pipeline.

The paper applies PCA to 63-feature data (before correlation-based filtering)
and reduces to 10 principal components for comparison with the full-feature
baseline.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PCA wrapper
# ---------------------------------------------------------------------------

def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Fit PCA on training data and transform both splits.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled training features.
    X_test : pd.DataFrame
        Scaled test features.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    X_train_pca, X_test_pca : pd.DataFrame
        Reduced-dimension feature matrices.
    pca : PCA
        Fitted PCA object (use .explained_variance_ratio_ for plotting).
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train)

    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]
    X_train_pca = pd.DataFrame(
        pca.transform(X_train), index=X_train.index, columns=pc_cols
    )
    X_test_pca = pd.DataFrame(
        pca.transform(X_test), index=X_test.index, columns=pc_cols
    )

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(
        f"PCA: {n_components} components explain "
        f"{cumulative_variance[-1] * 100:.1f}% of total variance"
    )
    return X_train_pca, X_test_pca, pca


def pca_variance_summary(pca: PCA) -> pd.DataFrame:
    """
    Return a tidy dataframe of explained variance per component.

    Parameters
    ----------
    pca : PCA
        A fitted sklearn PCA object.

    Returns
    -------
    pd.DataFrame
        Columns: component, explained_variance_ratio, cumulative_variance.
    """
    n = len(pca.explained_variance_ratio_)
    return pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, n + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
