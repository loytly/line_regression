import numpy as np
from sklearn.datasets import load_iris


def balanced_winnow_init(n_features, alpha=2.0, theta=None):
    """初始化权重与超参"""
    w_plus = np.ones(n_features, dtype=np.float64)
    w_minus = np.ones(n_features, dtype=np.float64)
    if theta is None:
        theta = n_features
    return w_plus, w_minus, alpha, theta


def balanced_winnow_predict(x, w_plus, w_minus, theta):
    """单样本预测，返回0或1"""
    score = np.dot(w_plus, x) - np.dot(w_minus, x)
    return 1 if score > theta else 0


def balanced_winnow_update(x, y, w_plus, w_minus, alpha, theta):
    """
    在线单步更新
    返回：(w_plus_new, w_minus_new, is_error)
    """
    y_hat = balanced_winnow_predict(x, w_plus, w_minus, theta)
    if y_hat == y:
        # 预测正确，权重不变
        return w_plus.copy(), w_minus.copy(), False

    wp = w_plus.copy()
    wm = w_minus.copy()
    mask = (x == 1)

    if y == 1:
        # 假阴性：放大正向权重，缩小负向权重
        wp[mask] *= alpha
        wm[mask] /= alpha
    else:
        # 假阳性：缩小正向权重，放大负向权重
        wp[mask] /= alpha
        wm[mask] *= alpha

    return wp, wm, True


def balanced_winnow_stream(X, y, n_features, alpha=2.0, theta=None, epochs=10):
    """流式遍历整个数据集训练，支持多轮迭代"""
    wp, wm, alpha, theta = balanced_winnow_init(n_features, alpha, theta)
    total_err = 0
    for epoch in range(epochs):
        epoch_err = 0
        for xi, yi in zip(X, y):
            wp, wm, err = balanced_winnow_update(xi, yi, wp, wm, alpha, theta)
            if err:
                epoch_err += 1
        total_err += epoch_err
        if epoch_err == 0:
            break
    return wp, wm, total_err, theta


def multi_quantile_binarize(X, quantiles):
    """
    多分位数递进二值化：每个特征用多个分位数阈值转为二值特征
    """
    n_samples, n_features = X.shape
    parts = []
    for i in range(n_features):
        for q in quantiles:
            threshold = np.percentile(X[:, i], q)
            parts.append((X[:, i] > threshold).astype(int))
    return np.column_stack(parts)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # setosa=1, 其他=0

    # 多分位数递进二值化
    X_binary = multi_quantile_binarize(X, quantiles=[20, 40, 60, 80])

    n_feat = X_binary.shape[1]
    w_plus, w_minus, total_err, threshold = balanced_winnow_stream(
        X_binary, y, n_features=n_feat, alpha=2.0, theta=0.5, epochs=20
    )

    print("特征维度:", n_feat)
    print("总错误数:", total_err)
    print("w+ =", w_plus)
    print("w- =", w_minus)


