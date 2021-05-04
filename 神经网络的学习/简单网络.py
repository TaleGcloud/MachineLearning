import sys, os
import numpy as np
from func.function import softmax, cross_entropy_error
from gradient import numerical_gradient


# 定义一简单网络（输入2个数，输出3个数），提供按照正态分布随机生成W初始值，提供预测值方法，计算loss函数的方法
class simpleNet:                                    # 生成一个简单网络函数
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):                           # 预测下一值函数
        return np.dot(x, self.W)

    def loss(self, x, t):                           # loss函数
        z = self.predict(x)                         # 预测下一值
        y = softmax(z)                              # 经由softmax输出
        loss = cross_entropy_error(y, t)            # 通过交叉熵算loss
        return loss


net = simpleNet()                                   # 生成网络
x = np.array([0.6, 0.9])                            # 输入数据
t = np.array([0, 0, 1])                             # 输出数据


# 使用lambda定义简单函数
l = lambda w: net.loss(x, t)                        # 把net类的loss方法封装成一个函数


dw = numerical_gradient(l, net.W)                   # 计算梯度（函数，值），即对w求loss的偏导

# 通过mini-batch随机选一组数据进行梯度下降的方法叫“随机梯度下降法”用名为SGD的函数实现
