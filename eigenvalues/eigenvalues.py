import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.

    Args:
        matrix: list[list[float]] or np.ndarray

    Returns:
        np.ndarray: Sorted eigenvalues (complex dtype if needed),
                    or None for invalid/non-square input.
    """
    try:
        # Convert input to NumPy array
        mat = np.array(matrix, dtype=float)

        # Check if matrix is 2D and square
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            return None

        # Handle empty matrix
        if mat.size == 0:
            return np.array([], dtype=complex)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(mat)

        # Sort by real part, then imaginary part
        idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
        return eigenvalues[idx]

    except (ValueError, TypeError):
        return None