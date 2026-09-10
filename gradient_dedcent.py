import numpy as np


def error_function(theta, X, y):
    """代价函数的定义"""
    diff = np.dot(X, theta) - y
    # transpose转置矩阵
    return (1. / 2 * m) * np.dot(np.transpose(diff), diff)


def gradient_function(theta, X, y):
    """函数J的梯度定义"""
    diff = np.dot(X, theta) - y
    return (1. / m) * np.dot(np.transpose(X), diff)


def gradient_descent(X, y, alpha):
    """执行梯度下降"""
    m=X.shape[1]
    theta = np.array([1, 1]).reshape(m, 1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta


if __name__ == '__main__':
    # 数据集大小
    m = 20

    # 点的x坐标和虚拟值 (x0, x1)
    # 创建一个m行1列的全为一矩阵
    X0 = np.ones((m, 1))
    """numpy.arange([start,] stop, [step,] dtype=None)
    复制
    参数解释
    start: 起点值，默认为0。
    stop: 终点值（不包含在结果中）。
    step: 步长，默认为1。
    dtype: 结果数组的数据类型，默认为None，根据输入参数自动推导。"""
    X1 = np.arange(1, m + 1).reshape(m, 1)

    X = np.hstack((X0, X1))
    # 点的y坐标
    y = np.array([
        3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
        11, 13, 13, 16, 17, 18, 17, 19, 21
    ]).reshape(m, 1)

    # 学习率
    alpha = 0.01

    optimal = gradient_descent(X, y, alpha)
    print('最优解:', optimal)
    print('代价函数值:', error_function(optimal, X, y)[0, 0])
