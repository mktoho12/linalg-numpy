import numpy as np
from scipy.linalg import lu


def demonstrate_lu_decomposition():
    """
    Demonstrates LU decomposition using SciPy.
    Decomposes a matrix A into P, L, and U such that P @ A = L @ U.
    """
    print("--- LU Decomposition ---")
    A = np.array([[1, 2], [3, 3]])
    print("Original Matrix A:")
    print(A)
    print()

    # Perform LU decomposition
    # P is the permutation matrix, L is lower triangular, U is upper triangular
    P, L, U = lu(A)

    print(
        "Permutation Matrix P (P.T is the permutation matrix from the standard definition):"
    )
    print(P)
    print()

    print("Lower Triangular Matrix L:")
    print(L)
    print()

    print("Upper Triangular Matrix U:")
    print(U)
    print()

    # Verification: P.T @ L @ U should be equal to A
    # Note: scipy's lu returns a P such that P @ A = L @ U.
    # The more standard definition is A = P @ L @ U, where P is different.
    # We will verify the scipy definition: P @ A = L @ U
    print("Verification (P @ A):")
    print(P @ A)
    print()
    print("Verification (L @ U):")
    print(L @ U)
    print()

    is_close = np.allclose(P @ A, L @ U)
    print(f"Is P @ A approximately equal to L @ U? {is_close}")
    print("-" * 25)

    P, L, U = lu(np.array([[2, 4], [1, 3]]))
    print("L行列")
    print(L)
    print("U行列")
    print(U)

