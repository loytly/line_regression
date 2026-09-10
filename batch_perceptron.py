import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def batch_perceptron(X, y, lr=0.01, max_iter=1000, verbose=True):
    """
    批处理感知器算法
    """
    m, n = X.shape

    w = np.zeros(n)
    b = 0

    errors = []

    if verbose:
        print(f"批处理感知器训练开始，样本数={m}，特征数={n}")
        print(f"学习率={lr}，最大迭代次数={max_iter}")
        print()

    for iteration in range(max_iter):
        w_delta = np.zeros(n)
        b_delta = 0
        error_count = 0

        for i in range(m):
            z_i = np.dot(X[i], w) + b
            if y[i] * z_i <= 0:
                w_delta += y[i] * X[i]
                b_delta += y[i]
                error_count += 1

        errors.append(error_count)

        if verbose and (iteration + 1) % 50 == 0:
            print(f"迭代 {iteration + 1}: 错误分类数 = {error_count}")

        if error_count == 0:
            if verbose:
                print(f"\n收敛！迭代次数: {iteration + 1}")
            break

        w = w + lr * w_delta
        b = b + lr * b_delta

    if error_count > 0 and verbose:
        print(f"\n达到最大迭代次数 {max_iter}，最终错误分类数: {error_count}")

    return w, b, np.array(errors)


def predict(X, w, b):
    """
    预测函数
    """
    z = np.dot(X, w) + b
    return np.sign(z)


def plot_decision_boundary(X, y, w, b, title="批处理感知器决策边界"):
    """
    绘制二维数据的决策边界
    """
    plt.figure(figsize=(10, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = predict(np.c_[xx.ravel(), yy.ravel()], w, b)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    unique_labels = np.unique(y)
    colors = ['red', 'blue']
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i],
                    label=f'类别 {label}', alpha=0.8, edgecolors='w')

    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_curve(errors, label="批处理感知器"):
    """
    绘制错误分类数变化曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, 'b-', linewidth=2, label=label)
    plt.xlabel('迭代次数')
    plt.ylabel('错误分类数')
    plt.title('批处理感知器训练错误曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("批处理感知器算法")
    print("=" * 60)
    print()

    print("【鸢尾花数据集（二分类）】")
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target[:]

    y = np.where(y == 0, -1, 1)

    print(f"数据集大小: {X.shape}")
    print(f"特征: {iris.feature_names[:2]}")
    print(f"类别分布: -1 (setosa): {np.sum(y == -1)}, 1 (versicolor/virginica): {np.sum(y == 1)}")
    print()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=" * 40)
    print("训练批处理感知器")
    print("=" * 40)
    w, b, errors = batch_perceptron(X_scaled, y, lr=0.1)

    print(f"\n训练完成")
    print(f"权重 w = {w}")
    print(f"偏置 b = {b}")

    predictions = predict(X_scaled, w, b)
    accuracy = np.mean(predictions == y)
    print(f"准确率: {accuracy * 100:.2f}%")

    print("\n" + "=" * 40)
    print("可视化决策边界")
    print("=" * 40)
    plot_decision_boundary(X_scaled, y, w, b, title="批处理感知器决策边界（鸢尾花二分类）")

    print("\n" + "=" * 40)
    print("错误分类曲线")
    print("=" * 40)
    plot_error_curve(errors)
