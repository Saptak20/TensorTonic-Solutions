def choose_threshold(y_true: list, scores: list, min_precision: float) -> dict:
    """
    Returns the threshold with the highest recall whose precision is at least min_precision.
    """
    total_pos = sum(y_true)
    n = len(scores)

    # Sort rows by score, highest first
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)

    best = None  # (recall, precision, threshold)
    tp = 0
    predicted = 0
    i = 0

    while i < n:
        t = scores[order[i]]
        # Include every row tied at this score (score >= t means all ties are predicted positive)
        while i < n and scores[order[i]] == t:
            tp += y_true[order[i]]
            predicted += 1
            i += 1

        precision = tp / predicted
        recall = tp / total_pos if total_pos > 0 else 0.0

        if precision >= min_precision:
            if (best is None
                    or recall > best[0]
                    or (recall == best[0] and precision > best[1])):
                best = (recall, precision, t)

    if best is None:
        return {"threshold": None, "precision": None, "recall": None}

    recall, precision, t = best
    return {"threshold": float(t), "precision": precision, "recall": recall}