import numpy as np


# 使用计算x附近左右偏移值的方法计算梯度
def numerical_gradient(f, x):                   # 计算梯度函数（输入函数（一般是Loss函数），x值）
    h = 1e-4                                    # 设置差量
    grad = np.zeros_like(x)                     # 创建和x形状相同的数组
    for idx in range(np.size(x, 0)):            # 获取x的个数
        tmp_val = x[idx]                        # 保存x索引值
        x[idx] = tmp_val + h                    # 右偏移
        fxh1 = f(x)                             # 计算右值
        x[idx] = tmp_val - h                    # 左偏移
        fxh2 = f(x)                             # 计算左值
        grad[idx] = (fxh1 - fxh2) / (2*h)       # 计算梯度
        x[idx] = tmp_val                        # 将保存的x索引值还原
    return grad
