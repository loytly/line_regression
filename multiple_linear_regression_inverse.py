import numpy as np
from sklearn.datasets import load_iris
import statsmodels.api as sm


def inverse_row(matrix):
    """检查矩阵"""
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        return None

    temp = 0
    row_temp = np.zeros(n * 2)

    identity = np.eye(n)
    mattemp = np.hstack((matrix.copy().astype(float), identity))

    i = 0
    while True:
        if mattemp[i][i] == 0:
            for k in range(i + 1, n):
                if mattemp[k][i] != 0:
                    temp1 = mattemp[i].copy()
                    mattemp[i] = mattemp[k]
                    mattemp[k] = temp1
                    break

        temp = mattemp[i][i]
        mattemp[i] = mattemp[i] / temp

        j = 0
        while True:
            if mattemp[j][i] != 0 and j != i:
                temp = -mattemp[j][i]
                row_temp = temp * mattemp[i]
                mattemp[j] = row_temp + mattemp[j]

            j += 1
            if j >= n:
                break

        i += 1
        if i >= n:
            break

    inverse_mat = mattemp[:, n:]
    return inverse_mat


def linear_regression(X, y):
    """
    θ = (X^T X)^(-1) X^T y
    """
    m, n = X.shape

    # 添加截距项（常数列1）
    X_intercept = sm.add_constant(X)
    print(X_intercept.T)
    # 显示特征矩阵
    print("特征矩阵 X（含截距项）")
    print(X_intercept)

    # 计算 X^T X
    XtX = np.dot(X_intercept.T, X_intercept)
    print(" 计算 X^T X")
    print(XtX)
    print()

    # 计算 X^T y
    Xty = np.dot(X_intercept.T, y)
    print("计算 X^T y")
    print(Xty)

    # 计算 (X^T X)^(-1)
    XtX_inverse = inverse_row(XtX)
    print("逆矩阵 (X^T X)^(-1):")
    print(XtX_inverse)

    # 计算 θ = (X^T X)^(-1) X^T y
    theta = np.dot(XtX_inverse, Xty)
    print("计算回归系数 θ = (X^T X)^(-1) X^T y")
    print(f"θ = {theta}")
    return theta


def predict(X, theta):
    """
    使用回归系数进行预测
    """
    m = X.shape[0]
    X_intercept = sm.add_constant(X)
    return np.dot(X_intercept, theta)


if __name__ == '__main__':
    iris = load_iris()
    x1 = iris.data[:, :1]
    x2 = iris.data[:, [1]]
    X = np.hstack((x1, x2))

    Y = iris.data[:, [2]]
    # X = np.array([
    #     [1, 2],
    #     [2, 4],
    #     [3, 6],
    # ])
    # y = np.array([9, 17, 25])
    theta = linear_regression(X, Y)
    y_pred = predict(X, theta)
    print(f"预测值: {y_pred}")
    print(f"实际值: {Y}")
