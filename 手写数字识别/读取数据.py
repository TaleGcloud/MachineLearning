import sys, os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
sys.path.append(os.pardir)


def img_show(imge):
    pil_img = Image.fromarray(np.uint8(imge))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]            # 图像索引
label = t_train[0]          # 标签索引
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
