def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    max_lag = max(lags)

    result = []

    for t in range(max_lag, len(series)):
        row = []

        for lag in lags:
            row.append(series[t - lag])

        result.append(row)

    return result