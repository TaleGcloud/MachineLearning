import numpy as np
import matplotlib.pyplot as plt


def step_function(num):           # 通用阶跃函数
    return np.array(num > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
