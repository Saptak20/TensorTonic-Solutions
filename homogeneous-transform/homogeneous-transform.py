import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    matrix = np.array(T, dtype=float)
    points_array = np.array(points, dtype=float)

    is_single_point = points_array.ndim == 1
    if is_single_point:
        points_array = points_array.reshape(1, 3)

    ones_column = np.ones((points_array.shape[0], 1))
    homogeneous_points = np.hstack([points_array, ones_column])

    transformed = homogeneous_points @ matrix.T
    result = transformed[:, :3]

    if is_single_point:
        return result[0]
    return result