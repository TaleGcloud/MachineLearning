import sys, os
import numpy as np
from dataset.mnist import load_mnist
sys.path.append(os.pardir)


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

# mini-batch
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)       # 从60000个数据中随机选10个
x_batch = x_train[batch_mask]                               # 选出简短数据集
t_batch = t_train[batch_mask]                               # 选出简短数据集


def cross_entropy_error(y, t):                              # 计算交叉熵函数
    if y.ndim == 1:                                         # 如果y的维度是1
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t, np.log(y + 1e-7)) / batch_size
