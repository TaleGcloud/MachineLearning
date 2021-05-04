from dataset.mnist import load_mnist
import numpy as np
import pickle
import func.function as fun


def get_data():                                             # 读取数据函数
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():                                         # 读入已保存的网络函数
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):                                    # 计算预测值函数
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = fun.sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = fun.sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = fun.softmax(a3)
    return y


# 批处理可以设置一批的数量，一批一批的处理数据
x, t = get_data()                                           # 读取数据到x（测试图像），t（测试标签），共10000组数据
network = init_network()                                    # 读入神经网络到network

batch_size = 100                                            # 批数量
accuracy_cnt = 0                                            # 正确判断的数目

for i in range(0, len(x), batch_size):                      # 开始循环判断，每隔batch_size个选一个
    x_batch = x[i:i+batch_size]                             # 取出从i到i+batch_size的数据
    y_batch = predict(network, x_batch)                     # 计算网络预测值
    p = np.argmax(y_batch, axis=1)                          # 取出y中最大值的索引，axis=1沿着一维方向寻找
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accurary:" + str(float(accuracy_cnt) / len(x)))      # 打印正确率（结果为：93.52%的正确率）
