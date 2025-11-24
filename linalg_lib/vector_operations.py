import numpy as np


def demonstrate_vector_operations():
    """Demonstrates various vector operations using NumPy."""
    print("--- Vector Operations ---")

    # ベクトルを定義する
    v = np.array([1, 2, 3])
    print("ベクトル v:", v)

    w = np.array([1, 2, 3, 4, 5])
    print("ベクトル w:", w)
    print()

    # ベクトルの要素にアクセスする
    print("## Vector Indexing & Slicing ##")
    print("ベクトル v の最初の要素:", v[0])
    print("ベクトル w の3番目の要素:", w[2])

    # スライスを使って部分ベクトルを取得する
    print("ベクトル w の最初の3つの要素:", w[:3])
    print("ベクトル w の3番目から最後までの要素:", w[2:])
    print("ベクトル w の3番目から4番目の要素:", w[2:4])
    print()

    # ベクトルの足し算とスカラー倍
    print("## Vector Arithmetic ##")
    w_add = np.array([2, 3, 4])
    print("ベクトル v:", v)
    print("ベクトル w_add:", w_add)
    print("ベクトル v + ベクトル w_add:", v + w_add)
    print("ベクトル v の2倍:", 2 * v)
    print()

    # 配列の形を変える
    print("## Vector Reshaping ##")
    v_reshape = np.array([1, 2, 3])
    print("ベクトル v_reshape:", v_reshape)
    n = v_reshape.shape[0]
    print("ベクトル v_reshape の要素数:", n)
    col_vec = v_reshape.reshape(n, 1)
    print("ベクトル v_reshape を列ベクトルに変形:")
    print(col_vec)
    row_vec = col_vec.reshape(1, n)
    print("ベクトル v_reshape を行ベクトルに変形:")
    print(row_vec)
    print()

    # ベクトルの転置、内積、ノルム
    print("## Transpose, Inner Product, Norm ##")
    print("行ベクトル row_vec:", row_vec)
    print("行ベクトル row_vec の転置:")
    print(row_vec.T)
    print("行ベクトル row_vec とその転置の内積:", row_vec @ row_vec.T)
    print("行ベクトル row_vec のノルム:", np.linalg.norm(row_vec))
    print("-" * 25 + "\n")


