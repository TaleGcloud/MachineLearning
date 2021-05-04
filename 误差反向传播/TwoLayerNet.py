from 激活函数层 import *
from collections import OrderedDict


class TwoLayerNet:
    # 初始化，定义网络结构和参数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.random.randn(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.random.randn(output_size)
        # 初始化层
        self.layers = OrderedDict()                 # 有序字典，方便遍历
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):                           # 预测方法只能算到输出前一层（因为输出层所需参数不同）
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):                           # loss函数，计算到loss
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):                       # 评价方法
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)                             # 必须先进行forward才能计算backward，计算到loss

        # backward                                    计算所有参数对loss求的偏导
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()                            # 将layer倒置，进行backward，更新本层对下层的dout导数
        for layer in layers:
            dout = layer.backward(dout)

        grads = dict()                              # 从每一层中取出定义参数的梯度
        grads['W1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        return grads
