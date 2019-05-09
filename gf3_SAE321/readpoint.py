# -*- coding: utf-8 -*-
'''
fun:文本文件中包含点数据信息，一行一个样本，一列一个特征（或类别），共106个特征+2个类别标签
读取文本数据返回归一化后的x以及one-hot形式的y. 主要用于点数据的分类例如：SAE/DBN
time:2018-6-26
'''
import pandas as pd
import  numpy as np
from keras.utils import np_utils
from sklearn import preprocessing

def readpoint(path,classnum,fs):

    feature_size = fs    #特征个数
    # col_index = []
    # for i in range(feature_size):
    #     col_index.append('f%d' % i)
    # col_index.append('class2')  # 列索引---2级类
    # col_index.append('class1')  # 列索引----1级类
    # col_index.append('index')  # 列索引-----全局索引

    #print(data.iloc[:5,:5])
    if 'tr'  in path:  #训练集

        data = pd.read_csv(path, sep=',')
        rows = len(data)  # 求出一共多少行
        cols = data.columns.size
        print("源文件共有 %d 行" % rows)
        print("源文件共有 %d 列" % cols)
        #print(data.iloc[:5, :5])
        # data['class2'].replace([r"landslide", r"farea"], [0, 1], inplace=True)
        #print(data['class2'].value_counts())


        train = np.array(data)  # 转成数组
        #print(train.shape[1])
        np.random.shuffle(train)  # 随机打乱
        x_train = train[:, 0:fs]  # 取x  第0-105列
        y_train = train[:,fs]-1  # 取y    第106列--标签列

        # indexpoint = train[:, 108]  # 取y    第108列--标签列

        trainx = preprocessing.scale(x_train.astype('float'))  # 数据预处理---scale标准化   0.42   减去均值除以标准差
        trainy = np_utils.to_categorical(y_train, classnum)  # 将y转换成one-hot的形式
    else:#测试集和验证集
        data = pd.read_csv(path, sep=',')
        rows = len(data)  # 求出一共多少行
        cols = data.columns.size
        print("源文件共有 %d 行" % rows)
        print("源文件共有 %d 列" % cols)
        # data['class2'].replace([r"landslide", r"farea"], [0, 1], inplace=True)
        #print(data['class2'].value_counts())
        #print(data.iloc[:5, :5])

        # print(y_train[:3])
        train = np.array(data)  # 转成数组
        #print(train.shape[1])
        #np.random.shuffle(train)  # 随机打乱
        x_train = train[:, 0:fs]  # 取x  0-34列
        y_train =  train[:,fs]-1
        #data.to_csv(r'C:\Users\1\Desktop\data4\traindata_replace.csv', index=False)
        trainy = np_utils.to_categorical(y_train, classnum)  # 将y转换成one-hot的形式

        # indexpoint = train[:, 108]  # 取y    第108列--标签列

        trainx = preprocessing.scale(x_train.astype('float'))  # 数据预处理---scale标准化   0.42   减去均值除以标准差

        #最大最小化
        # min_max_scaler = preprocessing.MinMaxScaler()
        # trainx = min_max_scaler.fit_transform(x_train.astype('float'))

        #正则化
        #trainx = preprocessing.normalize(x_train.astype('float'), norm='l2')


    #print(data['class2'].unique())

    # 使用iloc()切片，可以用序号代替列标签；但是loc()不行，如果有列标签必须使用列标签
    # print(data.iloc[:5,103:107])
    # pre_num = 8000  # 每一个类别的个数
    # if 'te' in path:
    #     pre_num = 2000
    # for i in range(int(rows / pre_num)):
    #     data.loc[i * pre_num:(i + 1) * pre_num, ['class2']] = int(i)

    # return trainx,trainy,indexpoint
    return trainx,trainy,y_train
#valdata, valtarget, y_val = readpoint(r'C:\Users\1\Desktop\data4\test1.txt', 20, 106)  # y_train

# classes = data['class'].unique()     #数据共有几个类别
# for c in classes:
#     name = 'data' + c
#     data_seg = data[ data['class']==c ]
#     data_seg.to_csv(r'./segm'+"//" +name+".csv",index=False)       #index=false控制输出文件中不自动添加列序号

#data.columns = col_index      #重新为列索引赋值
#print (data.columns)        #列索引名称
# print (data.index)       #行索引名称


#print(data[cols-1][:10])