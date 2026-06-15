import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    # Handle empty input
    if not seqs:
        return np.empty((0, 0), dtype=int)

    # Determine maximum length
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    # Create output array filled with pad_value
    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    # Copy (or truncate) each sequence
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]

    return result