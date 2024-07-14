import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
import sys
import CharNet_Package as cp
import ChineseCharNet_Package as ccp
from CharNet_Package import TwoLayerNet
from CharNet_Package import Affine
from CharNet_Package import Relu
from CharNet_Package import SoftmaxWithLoss

from ChineseCharNet_Package import TwoLayerNet
from ChineseCharNet_Package import Affine
from ChineseCharNet_Package import Relu
from ChineseCharNet_Package import SoftmaxWithLoss


# 蓝色的先验信息
Blue = 138
Green = 63
Red = 23

# 阈值
THRESHOLD = 50

ANGLE = -45
MIN_AREA = 2000
LICENSE_WIDTH = 440
LICENSE_HIGH = 140
MAX_WIDTH = 640

# 结果展示
def cv_show(name,img):
    cv2.namedWindow(name,0)
    print(img.shape)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,help="path to input image") #image是变量名
args = vars(parser.parse_args())

#读取原图像
img = cv2.imread(args["image"])
cv_show('img',img)
img_hight, img_width = img.shape[:2]

#-------------------------------预处理--------------------------------------
def preProcess():
    pass
#对图像进行缩放处理
if img_width > MAX_WIDTH:
    resize_rate = MAX_WIDTH / img_width
    img = cv2.resize(img,(MAX_WIDTH,int(img_hight * resize_rate)),interpolation=cv2.INTER_AREA)
# cv_show('img_resize',img)

# 高斯平滑
img_aussian = cv2.GaussianBlur(img,(5,5),1)
# cv_show('img_aussian',img_aussian)

#中值滤波
img_median = cv2.medianBlur(img_aussian,3)
cv_show('img_median',img_median)

#------------------------------车牌定位-------------------------------------
def locPlate():
    pass
#分离R、G、B通道
img_B = cv2.split(img_median)[0]
img_G = cv2.split(img_median)[1]
img_R = cv2.split(img_median)[2]
# print(img_B)
# print(img_G)
# print(img_R)
for i in range(img_median.shape[:2][0]):
    for j in range(img_median.shape[:2][1]):
        if abs(img_B[i,j] - Blue) < THRESHOLD and abs(img_G[i,j] - Green) <THRESHOLD and abs(img_R[i,j] - Red) < THRESHOLD:
            img_median[i][j][0] = 0
            img_median[i][j][1] = 0
            img_median[i][j][2] = 0
        else:
            img_median[i][j][0] = 255
            img_median[i][j][1] = 255
            img_median[i][j][2] = 255
cv_show('img_median',img_median)

kernel = np.ones((3,3),np.uint8)
img_erosion = cv2.erode(img_median,kernel,iterations = 5) #腐蚀操作
cv_show('img_erosion',img_erosion)
img_dilate = cv2.dilate(img_erosion,kernel,iterations = 5) #膨胀操作
cv_show("img_dilate",img_dilate)

img1 = cv2.cvtColor(img_dilate,cv2.COLOR_RGB2GRAY)
# cv_show('img1',img1)
image, contours, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)):
#     print(contours[i])
car_contours = []
if len(contours) > 0:
    for cnt in contours:  # TODO：此处需要一个异常处理（因为有可能/0）
        # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）rect[0]：矩形中心点坐标；rect[1]：矩形的高和宽；rect[2]：矩形的旋转角度
        rect = cv2.minAreaRect(cnt)
        # print('宽高:',rect[1])
        area_width, area_height = rect[1]
        # 计算最小矩形的面积，初步筛选
        area = rect[1][0] * rect[1][1]  # 最小矩形的面积
        if area > MIN_AREA:
            # 选择宽大于高的区域
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # print('宽高比：',wh_ratio)
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if wh_ratio > 2 and wh_ratio < 5.5:
                car_contours.append(rect)  # rect是minAreaRect的返回值，根据minAreaRect的返回值计算矩形的四个点
                box = cv2.boxPoints(rect)  # box里面放的是最小矩形的四个顶点坐标
                box = np.int0(box)  # 取整
                # for i in range(len(box)):
                #     print('最小矩形的四个点坐标：', box[i])
                # 获取四个顶点坐标
                left_point_x = np.min(box[:, 0])
                right_point_x = np.max(box[:, 0])
                top_point_y = np.min(box[:, 1])
                bottom_point_y = np.max(box[:, 1])
                left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
                right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
                top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
                bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
                vertices = np.array(
                    [[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],
                     [right_point_x, right_point_y]])
                oldimg = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                # print(car_contours)
                cv_show('oldimg', oldimg)
else:
    print("该图像中不存在车牌")

#-------------------------------车牌矫正-----------------------------------
for rect in car_contours:
    rect = (rect[0], (rect[1][0]+20, rect[1][1]+5), rect[2])
    box = cv2.boxPoints(rect)
    #图像矫正   cv2.getAffineTransform(pos1,pos2)，其中两个位置就是变换前后的对应位置关系。输出的就是仿射矩阵M，最后这个矩阵会被传给函数 cv2.warpAffine() 来实现仿射变换
    if rect[2] > ANGLE: #正角度
        new_right_point_x = vertices[0, 0]
        new_right_point_y = int(vertices[1, 1] - (vertices[0, 0] - vertices[1, 0]) / (vertices[3, 0] - vertices[1, 0]) * (vertices[1, 1] - vertices[3, 1]))
        new_left_point_x = vertices[1, 0]
        new_left_point_y = int(vertices[0, 1] + (vertices[0, 0] - vertices[1, 0]) / (vertices[0, 0] - vertices[2, 0]) * (vertices[2, 1] - vertices[0, 1]))
        point_set_1 = np.float32([[440, 0], [0, 0], [0, 140], [440, 140]])
    elif rect[2] < ANGLE: #负角度
        new_right_point_x = vertices[1, 0]
        new_right_point_y = int(vertices[0, 1] + (vertices[1, 0] - vertices[0, 0]) / (vertices[3, 0] - vertices[0, 0]) * (vertices[3, 1] - vertices[0, 1]))
        new_left_point_x = vertices[0, 0]
        new_left_point_y = int(vertices[1, 1] - (vertices[1, 0] - vertices[0, 0]) / (vertices[1, 0] - vertices[2, 0]) * (vertices[1, 1] - vertices[2, 1]))
        point_set_1 = np.float32([[0, 0], [0, 140], [440, 140], [440, 0]])
    new_box = np.array([(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]),(new_right_point_x, new_right_point_y)])
    point_set_0 = np.float32(new_box)
    mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
    dst = cv2.warpPerspective(img, mat, (440, 140))
    cv_show('dst',dst)

#-------------------------------字符分割-------------------------------------
plate_original = dst.copy()
img_aussian = cv2.GaussianBlur(dst,(5,5),1)
# cv_show('img_aussian',img_aussian)
#中值滤波
dst = cv2.medianBlur(img_aussian,3)

# 对车牌进行精准定位
img_B = cv2.split(dst)[0]
img_G = cv2.split(dst)[1]
img_R = cv2.split(dst)[2]
for i in range(dst.shape[:2][0]):
    for j in range(dst.shape[:2][1]):
        if abs(img_B[i,j] - Blue) < THRESHOLD and abs(img_G[i,j] - Green) <THRESHOLD and abs(img_R[i,j] - Red) < THRESHOLD:
            dst[i][j][0] = 0
            dst[i][j][1] = 0
            dst[i][j][2] = 0
        else:
            dst[i][j][0] = 255
            dst[i][j][1] = 255
            dst[i][j][2] = 255
# cv_show('dst',dst)

# 灰度化
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

#-------------------------------跳变次数去掉铆钉和边框----------------------------------
times_row = []  #存储哪些行符合跳变次数的阈值
for row in range(LICENSE_HIGH): # 按行检测 白字黑底
    pc = 0
    for col in range(LICENSE_WIDTH):
        if col != LICENSE_WIDTH-1:
            if gray[row][col+1] != gray[row][col]:
                pc = pc + 1
    times_row.append(pc)
print("每行的跳变次数:",times_row)

#找车牌的下边缘-从下往上扫描
row_end = 0
row_start = 0
for row in range(LICENSE_HIGH-2):
    if times_row[row] < 16:
        continue
    elif times_row[row+1] < 16:
        continue
    elif times_row[row+2] < 16:
        continue
    else:
        row_end = row + 2
print("row_end",row_end)

#找车牌的上边缘-从上往下扫描
i = LICENSE_HIGH-1
row_num = [] #记录row_start可能的位置
while i > 1:
    if times_row[i] < 16:
        i = i - 1
        continue
    elif times_row[i-1] < 16:
        i = i - 1
        continue
    elif times_row[i-2] < 16:
        i = i - 1
        continue
    else:
        row_start = i - 2
        row_num.append(row_start)
        i = i - 1
print("row_num",row_num)

#确定row_start最终位置
for i in range(len(row_num)):
    if i != len(row_num)-1:
        if abs(row_num[i] - row_num[i+1])>3:
            row_start = row_num[i]
print("row_start",row_start)

times_col = [0]
for col in range(LICENSE_WIDTH):
    pc = 0
    for row in range(LICENSE_HIGH):
        if row != LICENSE_HIGH-1:
            if gray[row,col] != gray[row+1,col]:
                pc = pc + 1
    times_col.append(pc)
print("每列的跳变次数",times_col)
# 找车牌的左右边缘-从左到右扫描
col_start = 0
col_end = 0
for col in range(len(times_col)):
    if times_col[col] > 2:
        col_end = col
print('col_end',col_end)

j = LICENSE_WIDTH-1
while j >= 0:
    if times_col[j] > 2:
        col_start = j
    j = j-1
print('col_start',col_start)

# 将车牌非字符区域变成纯黑色
for i in range(LICENSE_HIGH):
    if i > row_end or i < row_start:
        gray[i] = 0
for j in range(LICENSE_WIDTH):
    if j < col_start or j > col_end:
        gray[:,j] = 0
cv_show("res",gray)
# plate_binary = gray.copy()
for i in range(LICENSE_WIDTH-1,LICENSE_WIDTH):
    gray[:,i] = 0

# 字符细化操作
specify = cv2.erode(gray,kernel,iterations=2)
cv_show("specify",specify)
plate_specify = specify.copy()

#---------------------------垂直投影法切割字符-------------------------
lst_heise = []  #记录每一列中的白色像素点数量
for i in range(LICENSE_WIDTH):
    pc = 0
    for j in range(LICENSE_HIGH):
        if specify[j][i] == 255:
            pc = pc + 1
    lst_heise.append(pc)
# print("lst_heise",lst_heise)
a = [0 for i in range(0,LICENSE_WIDTH)]
for j in range(0, LICENSE_WIDTH):  # 遍历一列
    for i in range(0, LICENSE_HIGH):  # 遍历一行
        if specify[i, j] == 255:  # 如果该点为白点
            a[j] += 1  # 该列的计数器加一计数
            specify[i, j] = 0  # 记录完后将其变为黑色
    # print (j)
for j in range(0, LICENSE_WIDTH):  # 遍历每一列
    for i in range((LICENSE_HIGH - a[j]), LICENSE_HIGH):  # 从该列应该变白的最顶部的点开始向最底部涂白
        specify[i, j] = 255
plt.imshow(specify,cmap=plt.gray())
plt.show()
cv_show("touying",specify)

#开始找字符的边界
in_block = False #用来指示是否遍历到字符区
startIndex = 0
endIndex = 0
threshold = 10
index = 0
char_Image = [] # 存放一个个分割后的字符
for i in range(LICENSE_WIDTH):
    if lst_heise[i] != 0 and in_block == False: # 表示进入有白色像素点的区域
        in_block = True
        startIndex = i
        print("start", startIndex)
    elif lst_heise[i] == 0 and in_block == True: # 表示进入纯黑区域,且纯黑区域前面是字符区域
        endIndex = i
        in_block = False
        print("end", endIndex)
        if endIndex < startIndex:
            endIndex = 440
        if endIndex - startIndex > 10:
            res = plate_specify[row_start:row_end,startIndex:endIndex]
            index = index + 1
            res = cv2.resize(res, (20, 20), interpolation=cv2.INTER_LANCZOS4)  # 分割出的各个字符进行归一化
            char_Image.append(res)
            cv_show("res", res)

print("char_Image的长度:{}".format(len(char_Image)))
# print(char_Image)
char_Image = np.array(char_Image)
# print(char_Image.shape[0])
plate_char = np.hstack((char_Image[0],char_Image[1],char_Image[2],char_Image[3],
           char_Image[4],char_Image[5],char_Image[6]))
cv2.imshow("plate",plate_char)
cv2.waitKey(0)
cv2.destroyAllWindows()

#----------------------------------BP神经网络进行字符识别-------------------------------------
plate = [] # 保存车牌信息
for index,img in zip(range(char_Image.shape[0]),char_Image):
    img = img.flatten() / 255  # 归一化处理
    if index == 0:
        y,name = ccp.chinesePredict(img)
    else:
        y,name = cp.charPredict(img)
    plate.append(name)
print(plate)

