# -*- coding: utf-8 -*-
'''

'''
import tensorflow as tf
import numpy as np
import pandas as pd
import keras.backend.tensorflow_backend as KTF
# config = tf.ConfigProto(device_count={'gpu':0})
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# session = tf.Session(config=config)
# KTF.set_session(session)
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sys.path.append("../models")
sys.path.append("../base")
filename = os.path.basename(__file__)

from dbn import DBN
from cnn import CNN
from base_func import run_sess
from tensorflow.examples.tutorials.mnist import input_data
from sup_sae import supervised_sAE
from readpoint import readpoint
from ae import AE
import datetime
starttime = datetime.datetime.now()

# Loading dataset
# Each datapoint is a 8x8 image of a digit.
# mnist = input_data.read_data_sets('../dataset/MNIST_data', one_hot=True)
units=[
    # 100,200,300,400,500,
    #     1000
100
 ]

for i in range(1,501):
    # print("1到500")
    np.random.seed(1337)  # for reproducibility
    trainx, trainy, y_train = readpoint(r'D:\gao\jinzhou\33_doc\train.csv', 2, 196)  # trainx特征，trainy one-hot标签
    testx, testy, y_test = readpoint(r"D:\gao\jinzhou\fenge\te_33_"+str(i)+".csv", 2, 196)
    # valx,valy,y_val = readpoint(r'D:\gao\jiangxia\code1\val_2.txt',20,106)
    # Splitting data

    # datasets = [mnist.train.images,mnist.train.labels,mnist.test.images , mnist.test.labels]
    # datasets=[trainx,trainy,valx,valy]
    datasets1 = [trainx, trainy, testx, testy]
    # X_train, X_test = X_train[::100], X_test[::100]
    # Y_train, Y_test = Y_train[::100], Y_test[::100]
    # 验证
    # x_dim=datasets[0].shape[1]
    # y_dim=datasets[1].shape[1]
    # 测试
    x_dim = datasets1[0].shape[1]
    y_dim = datasets1[1].shape[1]
    p_dim = int(np.sqrt(x_dim))

    tf.reset_default_graph()
    # Training
    select_case = 3
    if select_case == 3:
        classifier = supervised_sAE(
            output_func='softmax',
            hidden_func='sigmoid',  # encoder：[sigmoid] | [affine]
            use_for='classification',
            loss_func='cross_entropy',  # decoder：[sigmoid] with ‘cross_entropy’ | [affine] with ‘mse’
            struct=[196, 2000, 2000, 2],
            lr=1e-3,
            epochs=1645,
            batch_size=1024,
            dropout=0.01,
            ae_type='ae',  # ae | dae | sae
            act_type=['sigmoid', 'affine'],
            noise_type='mn',  # Gaussian n oise (gs) | Masking noise (mn)
            beta=0.25,  # 惩罚因子权重（KL项 | 非噪声样本项）
            p=0.5,  # DAE：样本该维作为噪声的概率 / SAE稀疏性参数：期望的隐层平均活跃度（在训练批次上取平均）
            ae_lr=1e-3,
            ae_epochs=2000,
            pre_train=True)
    # run_sess(classifier,datasets,filename,load_saver='')#验证
    # print("***********************")
    run_sess(classifier, datasets1, filename, load_saver='f')  # 测试
    # label_distribution = classifier.label_distribution
    # data = pd.read_csv(r"D:\gao\jinzhou\code1\saver\label_distribution.csv")
    # data.to_csv(r"D:\gao\jinzhou\code1\saver\label_distribution_2_"+str(units[i])+"_"+str(random[j])+".csv",sep=",",index=False)
    hh = pd.read_csv(r"D:\gao\jinzhou\33_doc\predict_biaoqian_33_e.csv")
    hh.to_csv(r"D:\gao\jinzhou\33_doc\predict_biaoqian_33_"+ str(i) +".csv",sep=",",index=False)
    # hh.to_csv(r"D:\gao\jinzhou\ycbq\predict_biaoqian_1.csv",sep=",",index=False)

endtime = datetime.datetime.now()
print ((endtime - starttime).seconds)
