import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def lms(X, y, lr=0.1, iter_max=1000):
    m, n = X.shape
    w = np.zeros(n)
    error_curve = []

    for iter in range(iter_max):
        total_err = 0
        for i in range(m):
            # 计算误差
            e = y[i] - np.dot(X[i], w)

            # 更新权重
            w = w + lr * e * X[i]
            total_err += e ** 2
        mse = total_err / m
        error_curve.append(mse)

    return w, np.array(error_curve)


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data[:, :4]
    y = iris.target

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    w, err_curve = lms(X_scaled, y)
    print("收敛权重 w =", w)
    print("最终MSE:", err_curve[-1])
