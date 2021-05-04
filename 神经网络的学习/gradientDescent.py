from gradient import numerical_gradient
import numpy as np


# f需要优化的函数  init_x初始值   lr学习率   step_num学习次数
def gradient_descent(f, init_x, lr=0.01, step_num=100):             # gradient_descent函数
    x = init_x                                                      # 定义初始值
    for i in range(step_num):
        grad = numerical_gradient(f, x)                             # 调用梯度函数求梯度
        x -= lr*grad                                                # 梯度下降
    return x


def function(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function, init_x))
