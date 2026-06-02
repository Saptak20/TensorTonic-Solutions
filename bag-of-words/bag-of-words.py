import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    word_counts = {}

    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1

    return np.array([word_counts.get(word, 0) for word in vocab], dtype=int)