import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#from keras.utils import np_utils
from sklearn.cluster import KMeans
from scipy import signal
# from filter import hampel, smooth, Kalman1D, savgol
# from data_deal import peakRemoval
cut1=30
cut2=5
lin=130
lincut=130
linlong=162
ww=1
lin2=int((lin*2)/ww)
lincut2=int((lincut*2)/ww)
def read_data_cut(filenames):
    i = 0
    feature = []
    label = []
    label2 = []
    for filename in filenames:
        if os.path.exists(filename) == False:
            print(filename + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(filename, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        csvdata = csvdata[:, 0:270]
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2)-lincut ,
                                         int(csvdata.shape[0] / 2) +lincut, ww)])#取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature
        feat = np.sum(feat, axis=1)
        feat = np.rint(feat)
        a = np.argmax(feat)# 返回feature最大值位置
        idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lincut,a-cut1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a+cut1,int(temp_feature.shape[0] / 2) + lincut, ww)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        temp_feature = temp_feature[idx]
        #print(temp_feature)
        # 贴标签
        temp_label = -1  # 初始化
        temp_label2 = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3

        if ('zb' in filename):
            temp_label2 = 0
        elif ('zhw' in filename):
            temp_label2 = 1
        elif ('gzy' in filename):
            temp_label2 = 2
        elif ('lyx' in filename):
            temp_label2 = 3
        elif ('cyh' in filename):
            temp_label2 = 4
        elif ('ljc' in filename):
            temp_label2 = 5
        elif ('tk' in filename):
            temp_label2 = 6

        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        temp_label2 = np.tile(temp_label2, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            label2 = temp_label2
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
            label2 = np.concatenate((label2, temp_label2), axis=0)
    #label = np_utils.to_categorical(label)
    #label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label), np.array(label2)
def read_data(filenames):
    i = 0
    feature = []
    label = []
    label2 = []
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
        temp_label2 = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3

        if ('zb' in filename):
            temp_label2 = 0
        elif ('zhw' in filename):
            temp_label2 = 1
        elif ('gzy' in filename):
            temp_label2 = 2
        elif ('lyx' in filename):
            temp_label2 = 3
        elif ('cyh' in filename):
            temp_label2 = 4
        elif ('ljc' in filename):
            temp_label2 = 5
        elif ('tk' in filename):
            temp_label2 = 6
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        temp_label2 = np.tile(temp_label2, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            label2 = temp_label2
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
            label2 = np.concatenate((label2, temp_label2), axis=0)
    #label = np_utils.to_categorical(label)
    #label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label),np.array(label2)
def read_data_cutmid(filenames):
    i = 0
    feature = []
    label = []
    label2 = []
    for filename in filenames:
        if os.path.exists(filename) == False:
            print(filename + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(filename, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        csvdata = csvdata[:, 0:270]
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2)-linlong ,
                                         int(csvdata.shape[0] / 2) +linlong, ww)])#取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature
        feat = np.sum(feat, axis=1)
        feat = np.rint(feat)
        a = np.argmax(feat)# 返回feature最大值位置
        idx1 = np.array([j for j in range(a - lin, a - cut1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a + cut1, a + lin, ww)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        temp_feature = temp_feature[idx]
        #print(temp_feature)
        # 贴标签
        temp_label = -1  # 初始化
        temp_label2 = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3

        if ('zb' in filename):
            temp_label2 = 0
        elif ('zhw' in filename):
            temp_label2 = 1
        elif ('gzy' in filename):
            temp_label2 = 2
        elif ('tk' in filename):
            temp_label2 = 3
        elif ('cyh' in filename):
            temp_label2 = 4
        elif ('ljc' in filename):
            temp_label2 = 5
        elif ('lyx' in filename):
            temp_label2 = 6

        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        temp_label2 = np.tile(temp_label2, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            label2 = temp_label2
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
            label2 = np.concatenate((label2, temp_label2), axis=0)
    #label = np_utils.to_categorical(label)
    #label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label), np.array(label2)
def read_datamid(filenames):
    i = 0
    feature = []
    label = []
    label2 = []
    for filename in filenames:
        if os.path.exists(filename) == False:
            print(filename + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(filename, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        csvdata = csvdata[:, 0:270]
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2) - linlong,
                                         int(csvdata.shape[0] / 2) + linlong, ww)])#取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature
        feat = np.sum(feat, axis=1)
        feat = np.rint(feat)
        a = np.argmax(feat)  # 返回feature最大值位置
        idx1 = np.array([j for j in range(a - lin, a, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a, a + lin, ww)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        temp_feature = temp_feature[idx]
        # 贴标签
        temp_label = -1  # 初始化
        temp_label2 = -1  # 初始化
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3

        if ('zb' in filename):
            temp_label2 = 0
        elif ('zhw' in filename):
            temp_label2 = 1
        elif ('gzy' in filename):
            temp_label2 = 2
        elif ('tk' in filename):
            temp_label2 = 3
        elif ('cyh' in filename):
            temp_label2 = 4
        elif ('ljc' in filename):
            temp_label2 = 5
        elif ('lyx' in filename):
            temp_label2 = 6
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        temp_label2 = np.tile(temp_label2, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            label2 = temp_label2
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
            label2 = np.concatenate((label2, temp_label2), axis=0)
    #label = np_utils.to_categorical(label)
    #label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label),np.array(label2)
def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/caiji/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['zb','zhw', 'gzy', 'lyx', 'cyh', 'ljc','tk']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]

    trainfile += filenames[:140]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable,domain_label = read_data(trainfile)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    #print(kmeans.cluster_centers_.shape)
    #print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    #print(feature)
    k = np.arange(140)
    for i in range(0, 140):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:140]
    #np.random.shuffle(trainfile)

    for name in ['zb','zhw', 'gzy', 'lyx', 'cyh', 'ljc','tk']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile2 += filenames[:140]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable,domain_label = read_data(trainfile2)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    #print(kmeans.cluster_centers_.shape)
    #print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(140)
    for i in range(0, 140):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:140]
    #np.random.shuffle(trainfile2)

    testfile = trainfile[55:65]
    # trainfile = np.concatenate((trainfile[:55], trainfile[65:]), axis=0)
    # np.random.shuffle(trainfile)
    testfile2 = trainfile2[55:65]
    # trainfile2 = np.concatenate((trainfile2[:55], trainfile2[65:]), axis=0)
    # np.random.shuffle(trainfile2)

    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile
def other_file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/caiji/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for j in ["0"]:  # "1S", "2S"
        for i in [i for i in range(0, 20)]:
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable,domain_label = read_data(trainfile)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    #print(kmeans.cluster_centers_.shape)
    #print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(20)
    for i in range(0, 20):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:20]
    np.random.shuffle(trainfile)

    for j in ["1M"]:  # "1S", "2S"
        for i in [i for i in range(0, 20)]:
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile2 += filenames[:20]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable,domain_label = read_data(trainfile2)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    #print(kmeans.cluster_centers_.shape)
    #print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(20)
    for i in range(0, 20):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:20]
    np.random.shuffle(trainfile2)

    testfile = trainfile[10:]
    trainfile = trainfile[:20]
    testfile2 = trainfile2[10:]
    trainfile2 = trainfile2[:20]

    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile
def other_file_arraymid():
    filepath = 'D:/my bad/Suspicious object detection/data/caiji/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for j in ["0"]:  # "1S", "2S"
        for i in [i for i in range(0, 20)]:
            fn = filepath + "lyx-2.5-M/" + "lyx-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable,domain_label = read_datamid(trainfile)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(20)
    for i in range(0, 20):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:20]
    #np.random.shuffle(trainfile)

    for j in ["1M"]:  # "1S", "2S"
        for i in [i for i in range(0, 20)]:
            fn = filepath + "lyx-2.5-M/" + "lyx-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile2 += filenames[:20]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable,domain_label = read_datamid(trainfile2)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(20)
    for i in range(0, 20):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:20]
    #np.random.shuffle(trainfile2)

    testfile = trainfile[10:]
    trainfile = trainfile[:20]
    testfile2 = trainfile2[10:]
    trainfile2 = trainfile2[:20]

    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile


trainfile_array, testfile_array = file_array()#
train_feature, train_label,train_domain_label = read_datamid(trainfile_array)
train_feature_cut, train_label_cut,train_domain_label_cut = read_data_cutmid(trainfile_array)
test_feature, test_label,test_domain_label = read_datamid(testfile_array)

trainfile_other, testfile_other = other_file_arraymid()#
train_feature_ot, train_label_ot,train_domain_label_ot = read_datamid(trainfile_other)
train_feature_ot_cut, train_label_ot_cut,train_domain_label_ot_cut = read_data_cutmid(trainfile_other)
test_feature_ot, test_label_ot,test_domain_label_ot = read_datamid(testfile_other)
print(train_feature_ot.shape)
print(train_feature_ot_cut.shape)
# data1 = pd.read_csv('D:/my bad/Suspicious object detection/data/caiji/CSV/tk-2.5-M/tk-0-13.csv')
# data2 = pd.read_csv('D:/my bad/Suspicious object detection/data/caiji/CSV/tk-2.5-M/tk-0-14.csv')
# data3 = pd.read_csv('D:/my bad/Suspicious object detection/data/caiji/CSV/tk-2.5-M/tk-1M-13.csv')
# data4 = pd.read_csv('D:/my bad/Suspicious object detection/data/caiji/CSV/tk-2.5-M/tk-1M-14.csv')
# data1=data1[:350]
# data2=data2[:350]
# data3=data3[:350]
# data4=data4[:350]
# data3 = pd.read_csv('/home/zhw/实验部分/危险品/zhw-M-0.csv')w
# print(len(data1))
# print(len(data2))
# print(len(data3))
# data1 = data1.iloc[:, 120:121]  # 120～150是中间一对收发器的30个子载波数据23000:23100
# data2 = data2.iloc[:, 120:121]  # 120～150是中间一对收发器的30个子载波数据23000:23100
# data3 = data3.iloc[:, 120:121]  # 120～150是中间一对收发器的30个子载波数据23000:23100
# data4 = data4.iloc[:, 120:121]  # 120～150是中间一对收发器的30个子载波数据23000:23100
# data3 = data3.iloc[:, 120:150]  # 120～150是中间一对收发器的30个子载波数据23000:23100

t = range(lin2)
t2 = range(lincut2-cut1*2)
#
# dataMatrix1 = data1.values
# dataMatrix2 = data2.values
# dataMatrix3 = data3.values
# dataMatrix4 = data4.values
# print(dataMatrix1)
# print(dataMatrix1.shape)
# dataMatrix3 = data3.values

# for x in range(2, 3):
#     filtedData = hampel(dataMatrix[:, x])  # 首先经过hampel滤波
#     filtedData = smooth(filtedData, 25)  # 经过滑动平均滤波，去除高频噪声
#     filtedData = savgol(filtedData, 5, 3, 1)  # 这是S-G滤波，其为滑动滤波的升级版，画图可见其去除高频噪声效果更好
#     filtedData=np.array(filtedData)
#     plt.plot(t[:], filtedData[:])

# for x in range(3,9):
    # filtedData1 = hampel(dataMatrix1[0:200, x])  # 首先经过hampel滤波
    # filtedData1 = smooth(filtedData1, 5)  # 经过滑动平均滤波，去除高频噪声
    #
    # filtedData2 = hampel(dataMatrix2[0:200, x])  # 首先经过hampel滤波
    # filtedData2 = smooth(filtedData2, 5)  # 经过滑动平均滤波，去除高频噪声
    #
    # filtedData3 = hampel(dataMatrix3[0:200, x])  # 首先经过hampel滤波
    # filtedData3 = smooth(filtedData3, 5)  # 经过滑动平均滤波，去除高频噪声
    #
    # filtedData4 = hampel(dataMatrix4[0:200, x])  # 首先经过hampel滤波
    # filtedData4 = smooth(filtedData4, 5)  # 经过滑动平均滤波，去除高频噪声

# plt.plot(t[:], dataMatrix1, 'r')
# plt.plot(t[:], dataMatrix2, 'g')
# plt.plot(t[:], dataMatrix3, 'b')
# plt.plot(t[:], dataMatrix4, 'y')

k=70
#源域子载波波形图
# for i in range(0,140):
#     plt.plot(t[:], train_feature[i * lin2:(i + 1) * lin2,k:(k+1)], 'r')
#     plt.plot(t[:], train_feature[(i+50) * lin2:(i + 51) * lin2,k:(k+1)], 'b')
# plt.show()
# for i in range(0,140):
#     plt.plot(t2[:], train_feature_cut[i * (lincut2-cut1*2):(i + 1) * (lincut2-cut1*2),k:(k+1)], 'r')
#     plt.plot(t2[:], train_feature_cut[(i+50) * (lincut2-cut1*2):(i + 51) * (lincut2-cut1*2),k:(k+1)], 'b')
# plt.show()
#目标域子载波波形图
for i in range(0,20):
    plt.plot(t[:], train_feature_ot[i * lin2:(i + 1) * lin2,k:(k+1)], 'r')
    plt.plot(t[:], train_feature_ot[(i+15) * lin2:(i + 16) * lin2,k:(k+1)], 'b')
plt.show()
for i in range(0,20):
    plt.plot(t2[:], train_feature_ot_cut[i * (lincut2-cut1*2):(i + 1) * (lincut2-cut1*2),k:(k+1)], 'r')
    plt.plot(t2[:], train_feature_ot_cut[(i+15) * (lincut2-cut1*2):(i + 16) * (lincut2-cut1*2),k:(k+1)], 'b')
plt.show()