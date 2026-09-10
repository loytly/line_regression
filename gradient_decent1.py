import numpy as np
import matplotlib.pyplot as plt


# 梯度下降，输入x初始坐标，迭代次数，及学习率
def decent_fall(x, n, a):
    for i in range(n):
        cur_df = 2 * x - 2
        x = x - cur_df * a
        y = (x - 1) ** 2 + 1

        all_x.append(x)
        all_y.append(y)
    return


if __name__ == '__main__':
    # 用来存储x,y的所有坐标
    all_x = []
    all_y = []

    # 传参
    decent_fall(6, 10, 0.05)
    x = np.linspace(-5, 7, 100)
    y = (x - 1) ** 2 + 1

    # 把传入的x和y绘制成曲线
    plt.plot(x, y)
    # 坐标轴的范围，
    plt.axis([-7, 9, 0, 50])

    # 绘制散点图，并设置颜色
    plt.scatter(np.array(all_x), np.array(all_y), color='red')
    plt.show()

    print(all_x)
    print(all_y)
