def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    rows = len(X)
    cols = len(X[0])

    out_rows = rows // pool_size
    out_cols = cols // pool_size

    result = []

    for i in range(out_rows):
        row = []
        for j in range(out_cols):
            total = 0
            for a in range(pool_size):
                for b in range(pool_size):
                    total += X[i * pool_size + a][j * pool_size + b]
            row.append(total / (pool_size * pool_size))
        result.append(row)

    return result