import numpy as np


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


# 第1层
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])   # 权重，二维[上1层个数，下1层个数]
B1 = np.array([0.1, 0.2, 0.3])                      # 置偏，一维[下1层个数]
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

# 第2层
W2 = np.array([0.1, 0.4], [0.2, 0.5], [0.3, 0.6])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 第3层
W3 = np.array([0.1, 0.3], [0.2, 0.4])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3                            # 输出层 = Y
