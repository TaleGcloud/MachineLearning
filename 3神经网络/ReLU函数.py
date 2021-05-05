import numpy as np
import matplotlib.pyplot as plt


def relu(num):           # ReLU函数
    return np.maximum(0, num)


x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.1, 5.1)
plt.show()
