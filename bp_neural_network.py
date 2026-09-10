import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    f = sigmoid(z)
    return f * (1 - f)


def forward(x, w1, w2, b1, b2):
    z2 = np.dot(w1, x) + b1
    a2 = sigmoid(z2)
    z3 = np.dot(w2, a2) + b2
    a3 = sigmoid(z3)

    return z2, a2, z3, a3


def compute_cost(y_predict, y):
    m = y.shape[1]
    cost = np.sum((y - y_predict) ** 2) / (2 * m)
    return cost


def backward(x, y, z2, a2, z3, a3, w1, w2, b1, b2):
    m = x.shape[1]

    dz3 = (a3 - y) * sigmoid_derivative(z3)  # 输出层误差
    dw2 = np.dot(dz3, a2.T) / m
    db2 = np.sum(dz3, axis=1, keepdims=True) / m

    dz2 = np.dot(w2.T, dz3) * sigmoid_derivative(z2)
    dw1 = np.dot(dz2, x.T) / m
    db1 = np.sum(dz2, axis=1, keepdims=True) / m

    return dw1, dw2, db1, db2


def update_parameters(w1, w2, b1, b2, dw1, dw2, db1, db2, learning_rate):
    """更新偏置和权值"""
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    return w1, w2, b1, b2


def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * np.sqrt(2.0 / n_h)
    b2 = np.zeros((n_y, 1))

    return w1, w2, b1, b2


def train(X, Y, n_h, learning_rate=0.01, epochs=1000):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    w1, w2, b1, b2 = initialize_parameters(n_x, n_h, n_y)

    for epoch in range(epochs):
        z2, a2, z3, a3 = forward(X, w1, w2, b1, b2)

        cost = compute_cost(a3, Y)

        dw1, dw2, db1, db2 = backward(X, Y, z2, a2, z3, a3, w1, w2, b1, b2)

        w1, w2, b1, b2 = update_parameters(w1, w2, b1, b2, dw1, dw2, db1, db2, learning_rate)

        if (epoch + 1) % 1000 == 0:
            print(f"第 {epoch + 1} 轮: 代价 = {cost:.6f}")

    print(f"\n训练完成，最终代价: {cost:.6f}")

    return w1, w2, b1, b2


def predict(X, w1, w2, b1, b2):
    _, _, _, a3 = forward(X, w1, w2, b1, b2)
    return a3


if __name__ == '__main__':
    print("=" * 60)
    print("BP神经网络（鸢尾花分类）")
    print("=" * 60)
    print()

    iris = load_iris()
    x = iris.data[:, :4]
    y = iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    X = X_scaled.T  # 转置是让行为特征数，列为样本个数

    Y_onehot = np.eye(len(np.unique(y)))[y].T

    print(f"输入 X 形状: {X.shape}")
    print(f"输出 Y 形状: {Y_onehot.shape}")
    print()

    n_h = 4

    print(f"训练神经网络（隐藏层={n_h}个神经元）")
    print("-" * 30)
    w1, w2, b1, b2 = train(X, Y_onehot, n_h, learning_rate=1.0, epochs=30000)

    print("\n预测结果：")
    predictions = predict(X, w1, w2, b1, b2)
    pred_labels = np.argmax(predictions, axis=0)
    true_labels = y

    accuracy = np.mean(pred_labels == true_labels)
    print(f"预测类别:\n{pred_labels[:150]}")
    print(f"真实类别:\n{true_labels[:150]}")
    print(f"准确率: {accuracy:.2%}")
