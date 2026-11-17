import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    values = np.array(x, dtype=float)
    probabilities = np.array(p, dtype=float)

    total = np.sum(values * probabilities)

    return float(total)