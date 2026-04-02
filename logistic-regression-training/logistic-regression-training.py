import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X)
    y = np.array(y)

    n_samples, n_features = X.shape

    # Initialize
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        # Forward pass
        z = X @ w + b
        p = _sigmoid(z)

        # Gradients
        dw = (1 / n_samples) * (X.T @ (p - y))
        db = (1 / n_samples) * np.sum(p - y)

        # Update
        w -= lr * dw
        b -= lr * db

    return w, b