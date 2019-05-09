# -*- coding:utf-8 -*-
from osgeo import gdal
import numpy as np
import  pandas as pd
# import cv2
import matplotlib.pyplot as plt
from scipy import misc
import os
from PIL import Image

np.set_printoptions(threshold=np.inf)#使print大量数据不用符号...代替而显示所有

SIZE = 7   #截取的图片大小
data = pd.read_csv(r"E:\Python3\锦州\点\33_休耕地.txt")#行列号文件所在路径
rows = len(data )              #求出一共多少行
cols =  data.columns.size
print("源文件共有 %d 行"%rows)
print("源文件共有 %d 列"%cols)
print(data[:5])
data['x'] = round((data['X'] - 331718.588351)/1.9999961)                  #减去左
data['y'] = 21488-round((data['Y'] - 4516405.329)/1.9999961)           #减去下
print(data[:5])

#判断是否越界
# for i in range(rows ):
#     if data['x'][i]<32:
#         print(i)
#         print(data['x'][i])
# for j in range(rows ):
#     if data['y'][j]<32:
#         print(j)
#         print(data['y'][j])
 
dataset = gdal.Open(r"E:\Python3\锦州\锦州.tif")
cols=dataset.RasterXSize      #图像长度
rows=dataset.RasterYSize   #图像宽度
print('图像长'+ str(cols) +'个像素')
print('图像宽'+ str(rows) +'个像素')
#
band3 = dataset.GetRasterBand(3)#取第三波段
band2 = dataset.GetRasterBand(2)
band1 = dataset.GetRasterBand(1)
band4 = dataset.GetRasterBand(4)
#
x = data['x'] - (SIZE)/2
y = data['y'] - (SIZE)/2
count = 0
output=[]
for (i, j) in zip(x, y):

    i = float(i)
    j = float(j)
    if i>0 and j >0:
        b_1 = band1.ReadAsArray(i, j, SIZE, SIZE)# 以（ij）为中心位置，取size * size 的数据
        b_2 = band2.ReadAsArray(i, j, SIZE, SIZE)
        b_3= band3.ReadAsArray(i, j, SIZE, SIZE)
        b_4 = band4.ReadAsArray(i, j, SIZE, SIZE)
        b_1 = np.reshape(b_1,(1,49)) #将7*7数组拉伸为1*49
        b_2 = np.reshape(b_2, (1, 49))
        b_3 = np.reshape(b_3, (1, 49))
        b_4 = np.reshape(b_4, (1, 49))
        b = np.hstack((b_1,b_2,b_3,b_4))
    output.append(b)  #此时output的形状是（1000,1,196）
output = np.squeeze(output,axis=1) #np.squeeze只能消除了shape为1的维度
col_index = []
for i in range(196):
    col_index.append('%d' % i)
output = pd.DataFrame(output,columns=col_index)
output['class2']= 2
output['class1']=13
output.to_csv(r"E:\Python3\锦州\点\33_休耕地.csv",sep=",",index=False)
        # g = band2.ReadAsArray(i, j, SIZE, SIZE)
        # b = band1.ReadAsArray(i, j, SIZE, SIZE)
#数组横向合并
# output = np.hstack((output_1,output_2,output_3,output_4))
# print(output[:5])
# print(type(output))
# print(len(output))
# print(np.shape(output))



