def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    top_k_items = recommended[:k]
    relevant_set = set(relevant)

    hits = 0
    for item in top_k_items:
        if item in relevant_set:
            hits += 1

    precision = 0.0
    if k > 0:
        precision = hits / k

    recall = 0.0
    if len(relevant_set) > 0:
        recall = hits / len(relevant_set)

    return [float(precision), float(recall)]