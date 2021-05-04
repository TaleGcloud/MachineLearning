# 定义一个乘法层，提供前馈和反馈方法
class MulLayer:
    def __init__(self):         # 此乘法层只定义两个乘数
        self.x = None
        self.y = None

    def forward(self, x, y):    # 通过forward方法接收参数，并向前传播
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):   # 后通过backward方法反向传播求导
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


# 定义一个加法层， 提供前馈和反馈方法
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple, dapple_num, dorange, dorange_num, dtax)
