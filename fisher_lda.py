import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def fisher_lda(X, y, n_components=None, verbose=True):
    """
    Fisher线性判别分析（LDA）

    参数:
        X: 训练数据 (m, n)
        y: 训练标签 (m,)
        n_components: 降维后的维度，默认为None（自动计算为类别数-1）
        verbose: 是否打印过程信息

    返回:
        W: 投影矩阵 (n, n_components)
        means: 各类别的均值向量字典
        explained_variance_ratio: 解释方差比例
    """
    m, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    if n_components is None:
        n_components = min(n_classes - 1, n_features)

    if verbose:
        print(f"Fisher LDA训练开始")
        print(f"样本数: {m}, 特征数: {n_features}")
        print(f"类别数: {n_classes}, 目标维度: {n_components}")
        print()

    # 1. 计算各类别的均值向量
    class_means = {}
    for c in classes:
        class_means[c] = X[y == c].mean(axis=0)

    overall_mean = X.mean(axis=0)

    # 2. 计算类内散度矩阵 Sw
    Sw = np.zeros((n_features, n_features))
    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        # 计算每个类别的协方差矩阵
        Sc = np.zeros((n_features, n_features))
        for x in X_c:
            diff = x - class_means[c]
            Sc += np.outer(diff, diff)
        Sc /= n_c
        Sw += n_c * Sc

    # 3. 计算类间散度矩阵 Sb
    Sb = np.zeros((n_features, n_features))
    for c in classes:
        n_c = np.sum(y == c)
        mean_diff = class_means[c] - overall_mean
        Sb += n_c * np.outer(mean_diff, mean_diff)

    # 4. 求解广义特征值问题: Sw^(-1) * Sb * w = λ * w
    try:
        Sw_inv = np.linalg.inv(Sw)
        eig_values, eig_vectors = np.linalg.eigh(Sw_inv @ Sb)
    except np.linalg.LinAlgError:
        # 使用正则化
        Sw_reg = Sw + np.eye(n_features) * 1e-6
        Sw_inv = np.linalg.inv(Sw_reg)
        eig_values, eig_vectors = np.linalg.eigh(Sw_inv @ Sb)

    # 5. 选择最大的特征值对应的特征向量
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    # 选择前 n_components 个特征向量
    W = eig_vectors[:, :n_components]

    # 计算解释方差比例（相对于选定的主成分）
    selected_eig_values = eig_values[:n_components]
    explained_variance_ratio = selected_eig_values / np.sum(selected_eig_values)

    if verbose:
        print(f"特征值: {eig_values[:n_components]}")
        print(f"解释方差比例: {explained_variance_ratio}")
        print()

    means_dict = {c: class_means[c] for c in classes}

    return W, means_dict, explained_variance_ratio


def transform(X, W):
    """
    将数据投影到低维空间
    """
    return np.dot(X, W)


def plot_lda_projection_2d(X, y, W, title="Fisher LDA 二维投影"):
    """
    绘制LDA二维投影结果
    """
    X_transformed = transform(X, W)

    plt.figure(figsize=(10, 6))

    classes = np.unique(y)
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']

    for i, c in enumerate(classes):
        mask = y == c
        plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1],
                    c=colors[i], marker=markers[i],
                    label=f'类别 {c} ({iris.target_names[c] if c < len(iris.target_names) else c})',
                    alpha=0.7, edgecolors='w', s=50)

        # 绘制类别中心
        center = X_transformed[mask].mean(axis=0)
        plt.scatter(center[0], center[1], c=colors[i], marker='x',
                    s=200, linewidths=3)

    plt.title(title)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Fisher 线性判别分析 (LDA)")
    print("=" * 60)
    print()

    print("【鸢尾花数据集】")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"数据集大小: {X.shape}")
    print(f"特征: {iris.feature_names}")
    print(f"类别: {iris.target_names}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"类别分布: {dict(zip(unique, counts))}")
    print()

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=" * 40)
    print("完整数据训练Fisher LDA")
    print("=" * 40)

    # 使用全部数据（不分测试集和训练集）
    W, class_means, evr = fisher_lda(X_scaled, y, n_components=2)

    print(f"\n投影矩阵 W:")
    print(W)

    print("\n" + "=" * 40)
    print("投影到二维空间")
    print("=" * 40)
    X_transformed = transform(X_scaled, W)
    print(f"投影后数据形状: {X_transformed.shape}")

    print("\n" + "=" * 40)
    print("可视化结果")
    print("=" * 40)

    # 绘制二维投影
    plot_lda_projection_2d(X_scaled, y, W, title="Fisher LDA 鸢尾花数据集二维投影")


