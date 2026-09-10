import numpy as np
# 用于线性回归的类
from sklearn.linear_model import LinearRegression
# 用来切分训练集与测试集
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 设置输出精度。默认为8
np.set_printoptions(precision=2)

iris = load_iris()
# 获取花瓣长度作为x，花瓣宽度作为y。
x, y = iris.data[:, 2].reshape(-1, 1), iris.data[:, 3]

lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# 使用训练集数据，训练模型。
lr.fit(x_train, y_train)
print('权重：', lr.coef_)
print('截距：', lr.intercept_)
y_hat = lr.predict(x_test)
print('实际值：', y_test[:5])
print('预测值：', y_test[:5])

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 15

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, c='orange', label='训练集')
plt.scatter(x_test, y_test, c='g', marker='D', label='测试集')
plt.plot(x, lr.predict(x), 'r-')
plt.legend()
plt.xlabel('花瓣长度')
plt.ylabel('花瓣宽度')
