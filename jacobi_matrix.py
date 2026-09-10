import numpy as np


def max_value(A):
    n = A.shape[0]

    max_val = 0
    max_i = 0
    max_j = 1
    for i in range(n):
        for j in range(n):
            if i != j and abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                max_i = i
                max_j = j
    return max_i, max_j


def jacobi_eigen(A, eps=1e-10, max_iter=100):
    n = A.shape[0]

    """
    diag主要用于从一个一维数组中生成对角矩阵，或者从二维数组中提取对角线元素。
    内层提取对角线并返回一维数组，外层用数组生成对角矩阵。、

    """
    # 生成n维单位矩阵
    V = np.eye(n)
    for iteration in range(max_iter):
        max_i, max_j = max_value(A)
        p = max_i
        q = max_j
        if abs(A[p, q]) < eps:
            break
        if A[p][p] == A[q][q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan((2 * A[p, q]) / (A[p, p] - A[q, q]))

        # J是旋转矩阵
        P = np.eye(n)
        P[p, p] = np.cos(theta)
        P[q, q] = np.cos(theta)
        P[p, q] = -np.sin(theta)
        P[q, p] = np.sin(theta)

        A = np.dot(np.dot(P.T, A), P)
        V = np.dot(V, P)
    return np.diag(A), V


# 示例
if __name__ == '__main__':
    A = np.array([[4, -1, 1],
                  [-1, 3, -2],
                  [1, -2, 3]], dtype=float)
    eigvals, eigvecs = jacobi_eigen(A)
    print("特征值:", eigvals)
    print("特征向量:\n", eigvecs)
