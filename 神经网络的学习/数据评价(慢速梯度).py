import numpy as np
from dataset.mnist import load_mnist
from 两层神经网络 import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []                                                        # 训练loss列表
train_acc_list = []                                                             # 训练数据准确率列表
test_acc_list = []                                                              # 测试数据准确率列表

# 超参数
iters_num = 10000                                                           # 训练次数
train_size = x_train.shape[0]                                               # 训练的个数
batch_size = 100                                                            # 单批次个数
learning_rate = 0.1                                                         # 学习率
iter_per_epoch = max(train_size / batch_size, 1)                                # 更新次数

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)       # 定义（784， 50， 10）结构的两层网络

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)                   # 获取mini-batch索引
    x_batch = x_train[batch_mask]                                           # mini-batch训练数据输入
    t_batch = t_train[batch_mask]                                           # mini-batch训练数据输出

    gard = network.two_layer_gradient(x_batch, t_batch)                     # 计算梯度（此方法计算梯度太慢，后需改进）

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * gard[key]                    # 对每个参数梯度下降

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)                                            # 记录每次训练的loss值
    if i % iter_per_epoch == 0:                                                 # 每隔一段更新次数计算一次精度
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
