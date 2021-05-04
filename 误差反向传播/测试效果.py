import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet
from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000                                   # 训练次数
train_size = x_train.shape[0]                       # 总训练个数
batch_size = 100                                    # 批数量
learning_rate = 0.1                                 # 学习率
train_loss_list = []                                # 训练loss列表
train_acc_list = []                                 # 训练准确率列表
test_acc_list = []                                  # 测试准确率列表

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(i, loss)


def img_show(imge):
    pil_img = Image.fromarray(np.uint8(imge))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)


def set_train(num):
    img = x_train[num]            # 图像索引
    label = t_train[num]          # 标签索引
    print(label)

    img = img.reshape(28, 28)
    img_show(img)

    print(np.argmax(network.predict(x_train[num])))


pass
