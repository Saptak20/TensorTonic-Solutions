def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    n = len(values)

    if n == 0:
        return []

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return [counts[value] / n for value in values]