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


def demonstrate_qr_decomposition():
    """
    Demonstrates QR decomposition using NumPy.
    Decomposes a matrix A into an orthogonal matrix Q and an upper triangular matrix R.
    """
    print("\n--- QR Decomposition ---")
    A = np.array([[3,2],[4,1]])
    print("Original Matrix A:")
    print(A)
    print()

    # Perform QR decomposition
    Q, R = np.linalg.qr(A)

    print("Orthogonal Matrix Q:")
    print(Q)
    print()

    print("Upper Triangular Matrix R:")
    print(R)
    print()

    # Verification
    print("Verification (Q @ R):")
    reconstructed_A = Q @ R
    print(reconstructed_A)
    print()

    is_close = np.allclose(A, reconstructed_A)
    print(f"Is Q @ R approximately equal to A? {is_close}")
    print("-" * 25)
