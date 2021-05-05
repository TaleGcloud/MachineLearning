class Man:
    def __init__(self, name):                   # 构造函数
        self.name = name
        print("Initialized")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Goodbye " + self.name + "!")


m = Man("Tom")
m.hello()
m.goodbye()
