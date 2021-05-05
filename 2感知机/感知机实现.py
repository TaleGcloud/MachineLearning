import numpy as np


def my_and(x1, x2):             # 与门
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])    # 权重
    b = -0.7                    # 偏置
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1


def my_nand(x1, x2):            # 与非门
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def my_or(x1, x2):              # 或门
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def my_xor(x1, x2):             # 多层感知机
    s1 = my_nand(x1, x2)
    s2 = my_or(x1, x2)
    y = my_and(s1, s2)
    return y
