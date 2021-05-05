import numpy as np


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


def init_network():                 # 定义网络(保存每一层的权重和置偏)
    network = dict()                # 创建空字典
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def softmax(z3):                    # softmax函数
    return np.exp(z3) / np.sum(np.exp(z3))


def forward(network, x_hat):            # 前馈
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x_hat, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    z3 = np.dot(z2, w3) + b3
    y_hat = softmax(z3)

    return y_hat


net = init_network()
x = np.array([1.0, 0.5])
y = forward(net, x)
print(y)
