import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    if len(y) == 0:
        return 0.0

    labels = np.array(y)
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()

    entropy_value = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy_value)