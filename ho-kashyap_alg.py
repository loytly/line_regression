import numpy as np
from sklearn.datasets import load_iris


def ho_kashyap(Y, eta=0.5, max_iter=1000, tol=1e-4):
    """
    Ho-Kashyap算法
    Y : 规范化增广样本矩阵 N×d
    eta : 学习率(0<eta≤1)
    return a(权向量), b, flag(线性可分True/False)
    """
    N = Y.shape[0]
    b = np.ones(N)  # 初始化裕量向量>0
    Y_pinv = np.linalg.pinv(Y)

    for k in range(max_iter):
        a = Y_pinv @ b
        e = Y @ a - b
        e_plus = 0.5 * (e + np.abs(e))
        b = b + 2 * eta * e_plus

        # 收敛条件：误差全部非负，找到分离解
        if np.all(e >= -tol):
            return a, b, True
    # 达到最大迭代，判定不可分
    return a, b, False


# ==========测试示例==========
if __name__ == "__main__":
    # 构造线性可分样本
    iris = load_iris()
    Y = iris.data[:, :3]

    a, b, separable = ho_kashyap(Y, eta=0.4)
    print("权重向量 a =", a)
    print("是否线性可分：", separable)
    pred = Y @ a
    print("所有样本输出Ya：", pred)
