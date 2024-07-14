import os
import cv2
import numpy as np
import time
from collections import OrderedDict
import matplotlib.pylab as plt
from sklearn.externals import joblib

PATH = "E:\Train"
PROVINCES = ["zh_chuan", "川", "zh_e", "鄂", "zh_gan", "赣", "zh_gan1", "甘", "zh_gui", "贵", "zh_gui1", "桂", "zh_hei", "黑",
             "zh_hu", "沪", "zh_ji", "冀", "zh_jin", "津", "zh_jing", "京", "zh_jl", "吉", "zh_liao", "辽", "zh_lu", "鲁",
             "zh_meng", "蒙", "zh_min", "闽", "zh_ning", "宁", "zh_qing", "靑", "zh_qiong", "琼", "zh_shan", "陕", "zh_su",
             "苏", "zh_sx", "晋", "zh_wan", "皖", "zh_xiang", "湘", "zh_xin", "新", "zh_yu", "豫", "zh_yu1", "渝", "zh_yue",
             "粤", "zh_yun", "云", "zh_zang", "藏", "zh_zhe", "浙"]
DIR_NAME = "charsChinese"
# 最大迭代次数
ITEM_NUM = 10000
# 学习率
LEARNING_RATE = 0.1

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
    def forward(self,x):
        self.x = x
        out = np.dot(self.x,self.W) + self.b
        return out
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis = 0)
        return dx

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
def getData(path):
    chars_train = []
    chars_label = []
    list_label = []
    k = 1
    for root, dirs, files in os.walk(os.path.join(PATH, DIR_NAME)):
        if not os.path.basename(root).startswith("zh_"):  # startswith() 方法用于检查字符串是否是以指定子字符串开头
            continue
        pinyin = os.path.basename(root)
        # print("pinyin:{}".format(pinyin))
        index = PROVINCES.index(pinyin) + 1
        # print("index:{}".format(index))
        for filename in files:
            # print("input:{}".format(filename))
            filepath = os.path.join(root, filename)
            # print("filepath:{}".format(filepath))
            digit_img = cv2.imread(filepath)
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_RGB2GRAY)
            digit_img = digit_img.flatten()/255
            # print("digit_img形状:",digit_img.shape)
            chars_train.append(digit_img)
            chars_label.append(index)
    chars_train = np.array(chars_train)
    for i in range(len(chars_label)):
        single_label = [0] * 31
        if chars_label[i] == 2 * k - 1:
            single_label[k - 1] = 1
            list_label.append(single_label)
        if i != len(chars_label) - 1:
            if chars_label[i + 1] != 2 * k - 1:
                k = k + 1
    chars_label = np.array(list_label)
    # print(chars_train[0])
    # 将二维数组变为一维
    # print(chars_train[0].flatten())
    return chars_train,chars_label

def showImg(list_data):
    x = np.arange(0, len(list_data))
    y = np.array(list_data)
    plt.xlabel('x')
    plt.ylabel('accuracy')
    plt.plot(x, y)
    plt.show()

def train_ChineseNet():
    chars_train,chars_label = getData(PATH)
    # print("训练集形状",chars_train.shape)
    # print("标签形状",chars_label.shape)
    ccNet = TwoLayerNet(input_size = 400,hidden_size = 20,output_size = 31)
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
    # 训练准确度图
    showImg(train_acc_list)
    # 损失函数走势图
    # showImg(train_loss_list)
    print("程序运行时间:{}".format(end_time - begin_time))
    if not os.path.exists("chineseNet.pkl"):
        joblib.dump(ccNet,'chineseNet.pkl')
    else:
        os.remove("chineseNet.pkl")
        joblib.dump(ccNet,'chineseNet.pkl')

# 识别汉字
def chinesePredict(x):
    ccNet = joblib.load('chineseNet.pkl')
    y = ccNet.predict(x)
    index = np.argmax(y)
    print("汉字识别网络识别结果：{}".format(PROVINCES[index+1]))
    return y,PROVINCES[index+1]

if __name__ == "__main__":
    train_ChineseNet()
    print('汉字识别网络训练完成')