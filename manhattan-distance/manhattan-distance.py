import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    return float(np.sum(np.abs(x - y)))