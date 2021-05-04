from func.function import *


class ReLU:                                         # 定义ReLU激活层
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)                        # 将满足条件的保存为True
        out = x.copy()                              # 复制x到out
        out[self.mask] = 0                          # 将满足小于0的地方改为0
        return out

    def backward(self, dout):
        dout[self.mask] = 0                         # 满足小于0梯度为0，否则不变
        dx = dout
        return dx


class Sigmoid:                                      # 定义Sigmoid激活层
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = self.out * (1.0 - self.out) * dout     # 公式可得此计算梯度方法
        return dx


# Affine仿射层，即x通过加权W和b偏置的过程叫仿射层
class Affine:
    def __init__(self, w, b):
        self.w = w                                  # 仿射层权重，需初始化
        self.b = b                                  # 仿射层偏置，需初始化
        self.x = None                               # 输入数据
        self.dw = None                              # dw用于梯度下降
        self.db = None                              # db用于梯度下降

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)                 # 推导出的公式
        self.dw = np.dot(self.x.T, dout)            # 公式
        self.db = np.sum(dout, axis=0)              # 加法求导不变，维度上累加即可
        return dx


# 将softmax和loss一同合并
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None                            # loss
        self.y = None                               # softmax输出
        self.t = None                               # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]                # 测试数据批次大小
        dx = (self.y - self.t) / batch_size         # 除以批次大小，得到单个数据的误差
        return dx
