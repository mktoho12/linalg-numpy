import numpy as np
from numpy.typing import NDArray


def main() -> None:
    """
    Creates a 2x2 matrix, calculates its inverse, and prints both.
    """
    # Create a 2x2 NumPy array (matrix)
    matrix: NDArray[np.float64] = np.array([[1, 2], [3, 4]])

    print("Original Matrix:")
    print(matrix)
    print("-" * 20)

    try:
        # Calculate the inverse of the matrix
        inverse_matrix: NDArray[np.float64] = np.linalg.inv(matrix)
        print("Inverse Matrix:")
        print(inverse_matrix)
    except np.linalg.LinAlgError as e:
        print(f"Could not calculate the inverse matrix: {e}")


if __name__ == "__main__":
    main()
