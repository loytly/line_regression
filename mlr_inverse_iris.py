import numpy as np
from sklearn.datasets import load_iris


def compute_inverse(matrix):
    """
    使用行变换法求逆矩阵

    参数:
        matrix: 方阵 (n, n)

    返回:
        inverse: 逆矩阵 (n, n)
    """
    n = matrix.shape[0]
    mat_temp = np.hstack([matrix.astype(float), np.identity(n)])

    for i in range(n):
        if mat_temp[i, i] == 0:
            for k in range(i + 1, n):
                if mat_temp[k, i] != 0:
                    mat_temp[[i, k]] = mat_temp[[k, i]]
                    break

        pivot = mat_temp[i, i]
        mat_temp[i] = mat_temp[i] / pivot

        for j in range(n):
            if j != i and mat_temp[j, i] != 0:
                factor = -mat_temp[j, i]
                mat_temp[j] = mat_temp[j] + factor * mat_temp[i]

    return mat_temp[:, n:]


def multiple_linear_regression(X, y):
    """
    使用正规方程和逆矩阵求解多元线性回归系数

    参数:
        X: 特征矩阵 (m, n)，不含截距项
        y: 目标变量 (m,)

    返回:
        coefficients: 回归系数 (n+1,)，第一个为截距
    """
    m, n = X.shape
    X_with_intercept = np.hstack([np.ones((m, 1)), X.astype(float)])

    X_T = X_with_intercept.T
    X_T_X = X_T @ X_with_intercept
    X_T_y = X_T @ y

    X_T_X_inv = compute_inverse(X_T_X)

    coefficients = X_T_X_inv @ X_T_y

    return coefficients


def predict(X, coefficients):
    """
    使用回归系数进行预测

    参数:
        X: 特征矩阵 (m, n)，不含截距项
        coefficients: 回归系数 (n+1,)

    返回:
        predictions: 预测值 (m,)
    """
    m = X.shape[0]
    X_with_intercept = np.hstack([np.ones((m, 1)), X.astype(float)])
    return X_with_intercept @ coefficients


def compute_r_squared(y, y_pred):
    """
    计算决定系数 R²

    参数:
        y: 真实值 (m,)
        y_pred: 预测值 (m,)

    返回:
        r_squared: 决定系数
    """
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    return 1 - ss_residual / ss_total


if __name__ == '__main__':
    print("=" * 60)
    print("多元线性回归系数求解（逆矩阵方法）")
    print("=" * 60)
    print()

    print("加载鸢尾花数据集...")
    iris = load_iris()

    print("\n数据集信息:")
    print(f"样本数: {iris.data.shape[0]}")
    print(f"特征数: {iris.data.shape[1]}")
    print(f"特征名称: {iris.feature_names}")
    print(f"目标类别: {iris.target_names}")
    print()

    print("=" * 40)
    print("1. 使用花萼长度和花萼宽度预测花瓣长度")
    print("=" * 40)

    x1 = iris.data[:, :1]
    x2 = iris.data[:, [1]]
    X = np.hstack((x1, x2))
    y = iris.data[:, [2]].flatten()

    print(f"\n特征矩阵 X ({X.shape}):")
    print(f"特征 x1: 花萼长度")
    print(f"特征 x2: 花萼宽度")
    print(f"目标变量 y: 花瓣长度")
    print()

    coefficients = multiple_linear_regression(X, y)

    print("计算步骤:")
    print("-" * 30)
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
    X_T = X_with_intercept.T
    X_T_X = X_T @ X_with_intercept
    X_T_y = X_T @ y

    print(f"\n设计矩阵 X（含截距项）形状: {X_with_intercept.shape}")
    print(f"\nX^T X:\n{X_T_X}")
    print(f"\nX^T y:\n{X_T_y}")
    print(f"\n(X^T X)^(-1):\n{compute_inverse(X_T_X)}")
    print()

    print("回归系数:")
    print(f"截距项 b0: {coefficients[0]:.4f}")
    for i in range(1, len(coefficients)):
        print(f"系数 b{i}: {coefficients[i]:.4f}")

    print(f"\n回归方程:")
    equation = f"y = {coefficients[0]:.4f}"
    for i in range(1, len(coefficients)):
        sign = "+" if coefficients[i] >= 0 else "-"
        equation += f" {sign} {abs(coefficients[i]):.4f}*x{i}"
    print(equation)

    y_pred = predict(X, coefficients)
    r_squared = compute_r_squared(y, y_pred)
    print(f"\nR2 = {r_squared:.4f}")

    print("\n" + "=" * 40)
    print("2. 使用所有特征预测花萼长度")
    print("=" * 40)

    X_full = iris.data[:, 1:]
    y_full = iris.data[:, 0]

    print(f"\n特征矩阵 X ({X_full.shape}):")
    print(f"特征: 花萼宽度, 花瓣长度, 花瓣宽度")
    print(f"目标变量 y: 花萼长度")
    print()

    coefficients_full = multiple_linear_regression(X_full, y_full)

    print("回归系数:")
    print(f"截距项 b0: {coefficients_full[0]:.4f}")
    for i in range(1, len(coefficients_full)):
        print(f"系数 b{i}: {coefficients_full[i]:.4f}")

    print(f"\n回归方程:")
    equation_full = f"y = {coefficients_full[0]:.4f}"
    for i in range(1, len(coefficients_full)):
        sign = "+" if coefficients_full[i] >= 0 else "-"
        equation_full += f" {sign} {abs(coefficients_full[i]):.4f}*x{i}"
    print(equation_full)

    y_pred_full = predict(X_full, coefficients_full)
    r_squared_full = compute_r_squared(y_full, y_pred_full)
    print(f"\nR2 = {r_squared_full:.4f}")

    print("\n" + "=" * 60)
    print("多元线性回归分析完成")
    print("=" * 60)