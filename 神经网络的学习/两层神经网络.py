import sys, os
from func.function import *
from gradient import numerical_gradient
import numpy as np


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)      # 第1层权重
        self.params['b1'] = np.zeros(hidden_size)                                           # 第1层偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)     # 第2层权重
        self.params['b2'] = np.zeros(output_size)                                           # 第2层偏置

    def predict(self, x):                                                                   # 识别（推理）方法
        w1, w2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):                                                                   # 计算loss函数
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):                                                               # 计算识别精度
        y = self.predict(x)
        y = np.argmax(y, axis=1)                                    # 0维是组数，1维是每组每个值，故应比较1维
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def two_layer_gradient(self, x, t):                                                     # 计算所有参数梯度
        l = lambda w: self.loss(x, t)
        grads = dict()
        grads['W1'] = numerical_gradient(l, self.params['W1'])                              # 第1层权重梯度
        grads['b1'] = numerical_gradient(l, self.params['b1'])                              # 第1层偏置梯度
        grads['W2'] = numerical_gradient(l, self.params['W2'])                              # 第2层权重梯度
        grads['b2'] = numerical_gradient(l, self.params['b2'])                              # 第2层偏置梯度
        return grads
