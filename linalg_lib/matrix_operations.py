import numpy as np


def demonstrate_matrix_operations():
    """Demonstrates various matrix operations using NumPy."""
    print("--- Matrix Operations ---")

    # 行列を定義する
    A = np.array([[1, 2], [3, 3]])
    print("行列 A:")
    print(A)

    B = np.array([[1, 2, 3], [4, 5, 6]])
    print("行列 B:")
    print(B)
    print()

    # 行列の成分を取り出す
    print("## Matrix Indexing & Slicing ##")
    print("行列 A の1行1列の成分:", A[0, 0])
    print("行列 A の2行1列の成分:", A[1, 0])

    # スライス
    print("行列 A の1列目全体:", A[:, 0])
    print("行列 B の1行目全体:", B[0, :])
    print()

    # 行列のサイズと転置
    print("## Matrix Properties (Shape, Transpose) ##")
    print("行列 A のサイズ:", A.shape)
    print("行列 A の転置:")
    print(A.T)
    print("行列 B のサイズ:", B.shape)
    print("行列 B の転置:")
    print(B.T)
    print()

    # ベクトルと行列の掛け算
    print("## Matrix-Vector & Matrix-Matrix Multiplication ##")
    v = np.array([2, 3])
    print("行列 A と ベクトル v の積:", A @ v)

    # 行列と行列の掛け算
    L1 = np.array([[1, 0], [-3, 1]])
    print("行列 L1 と 行列 A の積:")
    print(L1 @ A)
    print()

    # 単位行列とゼロ行列
    print("## Special Matrices (Identity, Zeros) ##")
    E = np.eye(3)
    print("3x3の単位行列 E:")
    print(E)

    Z = np.zeros((2, 3))
    print("2x3のゼロ行列 Z:")
    print(Z)
    print()

    # 行列式
    print("## Determinant ##")
    det_A = np.linalg.det(A)
    print("行列 A の行列式:", det_A)
    print("-" * 25)
