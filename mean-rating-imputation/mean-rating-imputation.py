def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    rows = len(ratings_matrix)
    cols = len(ratings_matrix[0])

    result = [row[:] for row in ratings_matrix]

    if mode == "user":
        # Calculate mean for each user (row)
        for i in range(rows):
            ratings = [ratings_matrix[i][j] for j in range(cols)
                       if ratings_matrix[i][j] != 0]

            mean = sum(ratings) / len(ratings) if ratings else 0.0

            for j in range(cols):
                if ratings_matrix[i][j] == 0:
                    result[i][j] = mean

    elif mode == "item":
        # Calculate mean for each item (column)
        for j in range(cols):
            ratings = [ratings_matrix[i][j] for i in range(rows)
                       if ratings_matrix[i][j] != 0]

            mean = sum(ratings) / len(ratings) if ratings else 0.0

            for i in range(rows):
                if ratings_matrix[i][j] == 0:
                    result[i][j] = mean

    return result