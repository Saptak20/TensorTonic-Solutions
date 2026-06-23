import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.

    Args:
        X: list[list[float]] or np.ndarray of shape (N, D)

    Returns:
        np.ndarray of shape (D, D) containing the sample covariance matrix,
        or None if the input is invalid.
    """
    X = np.asarray(X, dtype=float)

    # Check if input is 2D and has at least 2 samples
    if X.ndim != 2 or X.shape[0] < 2:
        return None

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute sample covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    return cov_matrix