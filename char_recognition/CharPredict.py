import char_recognition.CharNet_Package as cp
import char_recognition.ChineseCharNet_Package as ccp
from sklearn.externals import joblib
import numpy as np

provinces = ccp.PROVINCES
char_class = cp.chars_class
#预测车牌，输出车牌号
# 识别字母和数字
def charPredict(x):
    charnet = joblib.load('charNet.pkl')
    # charnet = cp.TwoLayerNet(input_size = 400,hidden_size = 20,output_size = 34)
    y = charnet.predict(x)
    index = np.argmax(y)
    print("汉字识别网络识别结果：{}".format(provinces[index-1]))
    return y,provinces[index-1]
# 识别汉字

def chinesePredict(x):
    chinesenet = joblib.load('chineseNet.pkl')
    # chinesenet = ccp.TwoLayerNet(input_size = 400,hidden_size = 20,output_size = 31)
    y = chinesenet.predict(x)
    index = np.argmax(y)
    print("字符识别网络识别结果：{}".format(char_class[index]))
    return y,char_class[index]


