import numpy as np

def _entropy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    
    return float(-(p * np.log2(p)).sum())

def information_gain(y, split_mask):
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)

    H_parent = _entropy(y)

    y_left = y[split_mask]
    y_right = y[~split_mask]

    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    H_children = 0.0
    if n_left > 0:
        H_children += (n_left / n) * _entropy(y_left)
    if n_right > 0:
        H_children += (n_right / n) * _entropy(y_right)

    return H_parent - H_children