#带危险品的用一个aae重构，不带危险品的用另一个aae重构，重构数据比源数据多十倍
import pandas as pd
import os
from sklearn.cluster import KMeans
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils
import time
cut1=12
cut2=7
lin=162
ww=1
lin2=int((lin*2)/ww)
def read_data_cut1(filenames,kmeans1):
    i = 0
    feature = []
    label = []
    for filename in filenames:
        if os.path.exists(filename) == False:
            print(filename + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(filename, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        csvdata = csvdata[:, 0:270]
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2)-lin ,
                                         int(csvdata.shape[0] / 2) +lin, ww)])#取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature - kmeans1
        feat = np.square(feat)
        feat = np.sum(feat, axis=1)
        feat = np.sqrt(feat)
        a = np.argmax(feat)  # 返回feature最大值位置
        idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lin,a-cut1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a+cut1+2,int(temp_feature.shape[0] / 2) + lin, ww)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        temp_feature = temp_feature[idx]
        #print(temp_feature)
        # 贴标签
        temp_label = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
    #label = np_utils.to_categorical(label)
    return np.array(feature[:, :270]), np.array(label)
def read_data_cut2(filenames, kmeans2):
    i = 0
    feature = []
    label = []
    for filename in filenames:
        if os.path.exists(filename) == False:
            print(filename + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(filename, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        csvdata = csvdata[:, 0:270]
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2) - lin,
                                         int(csvdata.shape[0] / 2) + lin, ww)])  # 取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature - kmeans2
        feat = np.square(feat)
        feat = np.sum(feat, axis=1)
        feat = np.sqrt(feat)
        a = np.argmax(feat)  # 返回feature最大值位置
        idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lin, a-20-cut2+1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a-20+cut2+1, int(temp_feature.shape[0] / 2) + lin, ww)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        temp_feature = temp_feature[idx]
        # 贴标签
        temp_label = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
    #label = np_utils.to_categorical(label)
    return np.array(feature[:, :270]), np.array(label)
def read_data(filenames):
    i = 0
    feature = []
    label = []
    for filename in filenames:
        if os.path.exists(filename) == False:
            print(filename + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(filename, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        csvdata = csvdata[:, 0:270]
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2)-lin ,
                                         int(csvdata.shape[0] / 2) +lin, ww)])#取中心点处左右分布数据
        temp_feature = csvdata[idx,]
        # 贴标签
        temp_label = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
    label = np_utils.to_categorical(label)
    return np.array(feature[:, :270]), np.array(label)
def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/caiji/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['cyh','gzy','hsj','ljc','lyx','tk','zb','zhw']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]

    trainfile += filenames[:160]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable = read_data(trainfile)

    # k = np.arange(160)
    # feature, lable = read_data(trainfile)
    # feature = np.sum(feature, axis=1)
    # feature =np.rint(feature)
    # for i in range(0, 160):
    #     k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
    #     print("(不带东西特征)i为", end='')
    #     print(i)
    #     print(feature[i * lin2:(i + 1) * lin2])





    # kmeans = KMeans(n_clusters=1, n_init=50)
    # pred_train = kmeans.fit_predict(feature)
    #
    # #print(kmeans.cluster_centers_)
    # feature = feature - kmeans.cluster_centers_
    # feature = np.square(feature)
    # feature = np.sum(feature, axis=1)
    # feature = np.sqrt(feature)
    # #print(feature)
    # k = np.arange(90)
    # for i in range(0, 90):
    #     k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
    #     # print(k[i])
    # trainfile = trainfile[np.argsort(k)]
    # trainfile = trainfile[:75]
    # np.random.shuffle(trainfile)

    for name in ['cyh', 'gzy', 'hsj', 'ljc', 'lyx', 'tk', 'zb', 'zhw']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]

    trainfile2 += filenames[:160]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable = read_data(trainfile2)

    k = np.arange(160)
    feature, lable = read_data(trainfile2)
    feature = np.sum(feature, axis=1)
    feature = np.rint(feature)
    for i in range(0, 160):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        print("(带东西特征)i为", end='')
        print(i)
        print(feature[i * lin2:(i + 1) * lin2])

    # kmeans = KMeans(n_clusters=1, n_init=50)
    # pred_train = kmeans.fit_predict(feature)
    #
    # feature = feature - kmeans.cluster_centers_
    # feature = np.square(feature)
    # feature = np.sum(feature, axis=1)
    # feature = np.sqrt(feature)
    # k = np.arange(90)
    # for i in range(0, 90):
    #     k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
    #     # print(k[i])
    # trainfile2 = trainfile2[np.argsort(k)]
    # trainfile2 = trainfile2[:75]
    # # np.random.shuffle(trainfile2)
    #
    # testfile = trainfile[60:]
    # trainfile = trainfile[:75]
    # testfile2 = trainfile2[60:]
    # trainfile2 = trainfile2[:75]
    #
    # trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    # testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile
def other_file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for j in ["0"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile += filenames[:30]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable = read_data(trainfile)

    kmeans1 = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans1.fit_predict(feature)

    feature = feature - kmeans1.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(30)
    for i in range(0, 30):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:25]


    k = np.arange(25)
    feature, lable = read_data(trainfile)
    feature = np.sum(feature, axis=1)
    for i in range(0, 25):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        print("(不带东西特征)i为", end='')
        print(i)
        print(feature[i * lin2:(i + 1) * lin2])




    #np.random.shuffle(trainfile)

    # feature, lable = read_data(trainfile)
    # feature = feature - kmeans1.cluster_centers_
    # feature = np.square(feature)
    # feature = np.sum(feature, axis=1)
    # feature = np.sqrt(feature)
    # k = np.arange(25)
    # for i in range(0, 25):
    #     k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
    #     print("(不带东西特征)i为", end='')
    #     print(i)
    #     print(feature[i * lin2:(i + 1) * lin2])

    for j in ["1M"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile2 += filenames[:30]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable = read_data(trainfile2)

    kmeans2 = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans2.fit_predict(feature)

    feature = feature - kmeans2.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(30)
    for i in range(0, 30):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:25]

    k = np.arange(25)
    feature, lable = read_data(trainfile2)
    feature = np.sum(feature, axis=1)
    for i in range(0, 25):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        print("(带东西特征)i为", end='')
        print(i)
        print(feature[i * lin2:(i + 1) * lin2])
    #np.random.shuffle(trainfile2)

    # feature, lable = read_data(trainfile2)
    # feature = feature - kmeans2.cluster_centers_
    # feature = np.square(feature)
    # feature = np.sum(feature, axis=1)
    # feature = np.sqrt(feature)
    # k = np.arange(25)
    # for i in range(0, 25):
    #     k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
    #     print("(带东西特征)i为", end='')
    #     print(i)
    #     print(feature[i * lin2:(i + 1) * lin2])

    testfile = trainfile[20:]
    trainfile = trainfile[:25]
    testfile2 = trainfile2[20:]
    trainfile2 = trainfile2[:25]

    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile,kmeans1.cluster_centers_,kmeans2.cluster_centers_

trainfile_array, testfile_array = file_array()#

# train_feature, train_label = read_data(trainfile_array)
# test_feature, test_label = read_data(testfile_array)
#
# #trainfile_other, testfile_other,kmeans1,kmeans2= other_file_array()#
#
# train_feature_ot1, train_label_ot1 = read_data_cut1(trainfile_other[:25],kmeans1)
# train_feature_ot2, train_label_ot2 = read_data_cut2(trainfile_other[25:],kmeans2)
# train_feature_ot=np.concatenate((train_feature_ot1,train_feature_ot2), axis=0)
#
# train_label_ot=np.concatenate((train_label_ot1,train_label_ot2), axis=0)
# train_label_ot = np_utils.to_categorical(train_label_ot)
#
# test_feature_ot, test_label_ot = read_data(testfile_other)
# #全局归化为-1~1
# a=np.concatenate((train_feature, train_feature_ot), axis=0)
# print(train_feature.shape)
# print(train_feature_ot.shape)
# print(np.min(train_feature))
# print(np.max(train_feature))
# print(np.min(train_feature_ot))
# print(np.max(train_feature_ot))
#
# print(np.min(train_feature[:75*lin2]))
# print(np.max(train_feature[75*lin2:]))
# print(np.min(train_feature_ot1))
# print(np.max(train_feature_ot2))
# train_feature = ((train_feature.astype('float32')-np.min(a))-(np.max(a)-np.min(a))/2.0)/((np.max(a)-np.min(a))/2)
# test_feature = ((test_feature.astype('float32')-np.min(test_feature))-(np.max(test_feature)-np.min(test_feature))/2.0)/((np.max(test_feature)-np.min(test_feature))/2)
# train_feature_ot = ((train_feature_ot.astype('float32')-np.min(a))-(np.max(a)-np.min(a))/2.0)/((np.max(a)-np.min(a))/2)
# test_feature_ot = ((test_feature_ot.astype('float32')-np.min(test_feature_ot))-(np.max(test_feature_ot)-np.min(test_feature_ot))/2.0)/((np.max(test_feature_ot)-np.min(test_feature_ot))/2)
#
# X_train1 =train_feature[:75*lin2]
#
# X_test1 =test_feature[:5*lin2]
#
#
# X_train2 =train_feature[75*lin2:]
#
# X_test2 =test_feature[5*lin2:]


