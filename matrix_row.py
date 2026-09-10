import numpy as np


def in_matrix(m, n):
    matrix = np.zeros((m, n))
    for i in range(m):
        matrix[i] = list(map(int, input(f"输入第{i + 1}行的{n}个值").split()))
    print("输出矩阵")
    print(matrix)
    return matrix


def inverse_row(matrix):
    """检查矩阵"""
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        return None

    temp = 0
    row_temp = np.zeros(n * 2)

    # 生成单位矩阵
    identity = np.eye(n)
    # 转换矩阵格式并生成单位矩阵
    mattemp = np.hstack((matrix.copy().astype(float), identity))
    print(mattemp)

    i = 0
    while True:
        if mattemp[i][i] == 0:
            for k in range(i + 1, n):
                if mattemp[k][i] != 0:
                    temp1 = mattemp[i]
                    mattemp[i] = mattemp[k]
                    mattemp[k] = temp1

        temp = mattemp[i][i]
        mattemp[i] = mattemp[i] / temp
        row_temp = mattemp[i].copy()

        print(mattemp)

        j = 0
        while True:
            if mattemp[j][i] != 0 and j != i:
                temp = -mattemp[j][i]
                row_temp = temp * mattemp[i]
                mattemp[j] = row_temp + mattemp[j]

            j += 1
            if j >= n:
                break

        print(mattemp)

        i += 1
        if i >= n:
            break

    inverse_mat = mattemp[:, n:]
    print(inverse_mat)

    return inverse_mat


if __name__ == '__main__':
    print("=== 行变换矩阵求逆算法 ===")
    print("请输入方阵的维度 n:")
    n = int(input())

    print(f"\n请输入 {n}×{n} 矩阵:")
    A = in_matrix(n, n)

    inverse = inverse_row(A)
    print(inverse)
