def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == predicted_label:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1

    denominator = 2 * true_positives + false_positives + false_negatives

    if denominator == 0:
        return 0.0

    score = 2 * true_positives / denominator

    return round(float(score), 4)