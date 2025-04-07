import os
import cv2
import numpy as np
import time
from collections import OrderedDict
import matplotlib.pylab as plt
from sklearn.externals import joblib

PATH = "E:\Train"
CHARS_CLASS = ["0","1","2","3","4","5","6","7","8","9",
               "A","B","C","D","E","F","G",
               "H","J","K","L","M","N","P",
               "Q","R","S","T","U","V","W",
               "X","Y","Z"]
# 最大迭代次数
ITEM_NUM = 10000
# 学习率
LEARNING_RATE = 0.1

# 输出层激活函数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# 损失函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    # 前向传播
    def forward(self,x):
        self.x = x
        out = np.dot(self.x,self.W) + self.b
        return out
    # 后向反馈
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis = 0)
        return dx

#  隐藏层激活函数
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

# 读取输入数据
def getData(PATH):
    chars_train = []
    chars_label = []
    list_label = []
    k = 0
    for root, dirs, files in os.walk(os.path.join(PATH, "chars")):
        if len(os.path.basename(root)) > 1:  # startswith() 方法用于检查字符串是否是以指定子字符串开头
            continue
        index = CHARS_CLASS.index(os.path.basename(root)) # 获得种类对应的索引；basename函数: 返回path最后的文件名。如果path以／或\结尾，那么就会返回空值。即os.path.split(path)的第二个元素。
        for filename in files:
            filepath = os.path.join(root, filename)
            digit_img = cv2.imread(filepath)
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_RGB2GRAY)  #图片灰度化
            digit_img = digit_img.flatten() / 255  # 将图片转为一维向量
            chars_train.append(digit_img)
            chars_label.append(index)
    chars_train = np.array(chars_train)
    for i in range(len(chars_label)):
        single_label = [0] * 34
        if chars_label[i] == k:
            single_label[k] = 1
            list_label.append(single_label)
        if i != len(chars_label) - 1:
            if chars_label[i + 1] != k:
                k = k + 1
    chars_label = np.array(list_label)
    return chars_train,chars_label

# 显示图片
def showImg(list_data):
    x = np.arange(0, len(list_data))
    y = np.array(list_data)
    plt.xlabel('x')
    plt.ylabel('accuracy')
    plt.plot(x, y)
    plt.show()

# 训练字符识别网络
def train_CharNet():
    chars_train, chars_label = getData(PATH)
    ccNet = TwoLayerNet(input_size=400, hidden_size=20, output_size=34)
    # train_size = chars_train.shape[0]
    train_loss_list = []
    train_acc_list = []
    begin_time = time.time()
    for i in range(ITEM_NUM):
        # 通过误差反向传播法求梯度
        grad = ccNet.gradient(chars_train,chars_label)
        for key in ("W1","b1","W2","b2"):
            ccNet.params[key] -= LEARNING_RATE * grad[key]
        loss = ccNet.loss(chars_train,chars_label)
        train_loss_list.append(loss)
        train_acc = ccNet.accuracy(chars_train,chars_label)
        train_acc_list.append(train_acc)
        print(train_acc)
    end_time = time.time()
    # print(train_loss_list)
    # print(train_acc_list)
    # 显示训练准确度图
    showImg(train_acc_list)
    # 显示损失函数值
    # showImg(train_loss_list)
    print("程序运行时间:{}".format(end_time - begin_time))
    if not os.path.exists("charNet.pkl"):
        joblib.dump(ccNet,'charNet.pkl')
    else:
        os.remove("charNet.pkl")
        joblib.dump(ccNet,'charNet.pkl')

# 识别字符
def charPredict(x):
    ccNet = joblib.load('charNet.pkl')
    y = ccNet.predict(x)
    index = np.argmax(y)
    print("字符识别网络识别结果：{}".format(CHARS_CLASS[index]))
    return y,CHARS_CLASS[index]

if __name__ == "__main__":
    train_CharNet()
    print("字符识别网络训练完成")
