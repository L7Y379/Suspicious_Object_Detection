#带危险品的用一个aae重构，不带危险品的用另一个aae重构，重构数据比源数据多十倍
#latent_dim1 = 10  latent_dim2 = 64
#train_feature = ((train_feature.astype('float32')-np.min(a))-(np.max(a)-np.min(a))/2.0)/((np.max(a)-np.min(a))/2)
#classer  512 512 256 2
#dis 512  512  256 3
#加入域分类训练，将域无关信息往分类特征中转移
#进行分类训练时把encoder参数同样加入分类训练
import pandas as pd
import os
from sklearn.cluster import KMeans
from keras.datasets import mnist
from keras.layers import LSTM,Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
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
localtime1 = time.asctime( time.localtime(time.time()) )
print ("本地时间为 :", localtime1)
cut1=15
cut2_0=15
cut2_1M=15
lin=120
lincut=120
linlong=162
ww=1
lin2=int((lin*2)/ww)
lincut2=int((lincut*2)/ww)
nb_lstm_outputs = 50  #神经元个数
nb_time_steps = lin2-cut1*2  #时间序列长度
nb_input_vector = 270 #输入序列
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
        elif ('lyx' in filename):
            temp_label2 = 3
        elif ('cyh' in filename):
            temp_label2 = 4
        elif ('ljc' in filename):
            temp_label2 = 5
        elif ('tk' in filename):
            temp_label2 = 6

        temp_label3 = np.tile(temp_label, (1,))
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        temp_label2 = np.tile(temp_label2, (temp_feature.shape[0],))

        if i == 0:
            feature = temp_feature
            label = temp_label
            label2 = temp_label2
            label3 = temp_label3
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
            label2 = np.concatenate((label2, temp_label2), axis=0)
            label3 = np.concatenate((label3, temp_label3), axis=0)
    label = np_utils.to_categorical(label)
    label2 = np_utils.to_categorical(label2)
    label3 = np_utils.to_categorical(label3)
    return np.array(feature[:, :270]), np.array(label), np.array(label2),np.array(label3)
def read_data_cut2mid(filenames):
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
        if ('-0-' in filename):
            idx1 = np.array([j for j in range(a - lin, a - cut2_0, ww)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(a + cut2_0, a + lin, ww)])  # 取中心点处左右分布数据
            idx = np.hstack((idx1, idx2))
            temp_feature = temp_feature[idx]
        if ('-1M-' in filename):
            idx1 = np.array([j for j in range(a - lin, a - cut2_1M, ww)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(a + cut2_1M, a + lin, ww)])  # 取中心点处左右分布数据
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
        temp_label3 = np.tile(temp_label, (1,))
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        temp_label2 = np.tile(temp_label2, (temp_feature.shape[0],))
        if i == 0:
            feature = temp_feature
            label = temp_label
            label2 = temp_label2
            label3 = temp_label3
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接
            label = np.concatenate((label, temp_label), axis=0)
            label2 = np.concatenate((label2, temp_label2), axis=0)
            label3 = np.concatenate((label3, temp_label3), axis=0)
    label = np_utils.to_categorical(label)
    label2 = np_utils.to_categorical(label2)
    label3 = np_utils.to_categorical(label3)
    return np.array(feature[:, :270]), np.array(label), np.array(label2),np.array(label3)
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
        #print(filename)
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
    label = np_utils.to_categorical(label)
    label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label),np.array(label2)
def read_datamid_huodong():
    k=1
    fn1 = 'D:/my bad/Suspicious object detection/data/huodong/room1/data.csv'
    csvdata = pd.read_csv(fn1, header=None)
    csvdata = np.array(csvdata, dtype=np.float64)
    print(csvdata.shape)#(1200,18000)
    train_feature=csvdata.reshape(240000,90)
    train_feature_ot=train_feature[(k-1)*120*200:k*120*200]
    train_feature=np.concatenate((train_feature[:(k-1)*120*200], train_feature[k*120*200:]), axis=0)
    for i in range(54):
        idx1 = np.array([j for j in range((i * 20)*200, (i * 20 + 8)*200, 1)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range((i * 20 + 13)*200, (i * 20 + 20)*200, 1)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        idxt = np.array([j for j in range((i * 20 + 8)*200, (i * 20 + 13)*200, 1)])
        # print(filename)
        if i == 0:
            temp_feature = train_feature[idx]
            temp_feature2 = train_feature[idxt]
        else:
            temp_feature = np.concatenate((temp_feature, train_feature[idx]), axis=0)
            temp_feature2 = np.concatenate((temp_feature2, train_feature[idxt]), axis=0)
    train_feature = temp_feature
    test_feature = temp_feature2
    print(train_feature.shape)#(162000,90)
    print(test_feature.shape)#(54000,90)
    print(train_feature_ot.shape)#(24000,90)

    for i in range(9):
        temp_domain_label = np.tile(i, (18000,))
        if i==0:
            train_domain_label=temp_domain_label
        else:
            train_domain_label = np.concatenate((train_domain_label, temp_domain_label), axis=0)
    print(train_domain_label.shape)#(162000,)
    train_domain_label=np_utils.to_categorical(train_domain_label)
    print(train_domain_label.shape)#(162000,9)
    print(train_domain_label.shape)

    fn2 = 'D:/my bad/Suspicious object detection/data/huodong/room1/label.csv'
    csvdata = pd.read_csv(fn2, header=None)
    csvdata = np.array(csvdata, dtype=np.float64)
    print(csvdata.shape)  # (1200,6)
    temp_label = csvdata
    temp_label=temp_label.repeat(200,axis=0)
    # for i in range(60):
    #     print(temp_label[i*20*200:(i+1)*20*200])
    train_label_ot=temp_label[(k-1)*120*200:k*120*200]
    train_label = np.concatenate((temp_label[:(k-1)*120*200], temp_label[k*120*200:]), axis=0)
    for i in range(54):
        idx1 = np.array([j for j in range((i * 20)*200, (i * 20 + 8)*200, 1)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range((i * 20 + 13)*200, (i * 20 + 20)*200, 1)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        idxt = np.array([j for j in range((i * 20 + 8)*200, (i * 20 + 13)*200, 1)])
        # print(filename)
        if i == 0:
            temp_label = train_label[idx]
            temp_label2 = train_label[idxt]
        else:
            temp_label = np.concatenate((temp_label, train_label[idx]), axis=0)
            temp_label2 = np.concatenate((temp_label2, train_label[idxt]), axis=0)
    train_label = temp_label
    test_label = temp_label2
    print(train_label.shape)  # (810,6)
    print(test_label.shape)  # (270,6)
    print(train_label_ot.shape)  # (120,6)

    return np.array(train_feature), np.array(test_feature),np.array(train_feature_ot),np.array(train_domain_label),np.array(train_label),np.array(test_label),np.array(train_label_ot)
#对每个人的数据单独聚类
def file_array3():
    filepath = 'D:/my bad/Suspicious object detection/data/caiji/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['zb']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[17:18]
    trainfile = np.concatenate((trainfile[1:1], trainfile[1:16]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain = trainfile
    alltest = testfile
    trainfile = []

    for name in ['zhw']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:12]
    trainfile = np.concatenate((trainfile[:8], trainfile[12:19]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain = np.concatenate((alltrain, trainfile), axis=0)
    alltest = np.concatenate((alltest, testfile), axis=0)
    trainfile = []
    for name in ['gzy']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:10]
    trainfile = np.concatenate((trainfile[:8], trainfile[10:17]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain = np.concatenate((alltrain, trainfile), axis=0)
    alltest = np.concatenate((alltest, testfile), axis=0)
    trainfile = []
    for name in ['lyx']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:10]
    trainfile = np.concatenate((trainfile[:8], trainfile[10:17]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain = np.concatenate((alltrain, trainfile), axis=0)
    alltest = np.concatenate((alltest, testfile), axis=0)
    trainfile = []
    for name in ['cyh']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:11]
    trainfile = np.concatenate((trainfile[:8], trainfile[11:18]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain = np.concatenate((alltrain, trainfile), axis=0)
    alltest = np.concatenate((alltest, testfile), axis=0)
    trainfile = []
    for name in ['ljc']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:11]
    trainfile = np.concatenate((trainfile[:8], trainfile[11:18]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain = np.concatenate((alltrain, trainfile), axis=0)
    alltest = np.concatenate((alltest, testfile), axis=0)
    trainfile = []
    for name in ['zb']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[17:18]
    trainfile = np.concatenate((trainfile[1:1], trainfile[1:16]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain2 = trainfile
    alltest2 = testfile
    trainfile = []
    for name in ['zhw']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:12]
    trainfile = np.concatenate((trainfile[:8], trainfile[12:19]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain2 = np.concatenate((alltrain2, trainfile), axis=0)
    alltest2 = np.concatenate((alltest2, testfile), axis=0)
    trainfile = []
    for name in ['gzy']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:10]
    trainfile = np.concatenate((trainfile[:8], trainfile[10:17]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain2 = np.concatenate((alltrain2, trainfile), axis=0)
    alltest2 = np.concatenate((alltest2, testfile), axis=0)
    trainfile = []
    for name in ['lyx']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:10]
    trainfile = np.concatenate((trainfile[:8], trainfile[10:17]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain2 = np.concatenate((alltrain2, trainfile), axis=0)
    alltest2 = np.concatenate((alltest2, testfile), axis=0)
    trainfile = []
    for name in ['cyh']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(20)
    for i in range(0, 20):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile = trainfile[:20]
    testfile = trainfile[8:11]
    trainfile = np.concatenate((trainfile[:8], trainfile[11:18]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain2 = np.concatenate((alltrain2, trainfile), axis=0)
    alltest2 = np.concatenate((alltest2, testfile), axis=0)
    trainfile = []
    for name in ['ljc']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable, domain_label = read_datamid(trainfile)
    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
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
    testfile = trainfile[8:11]
    trainfile = np.concatenate((trainfile[:8], trainfile[11:18]), axis=0)
    # np.random.shuffle(trainfile)
    alltrain2 = np.concatenate((alltrain2, trainfile), axis=0)
    alltest2 = np.concatenate((alltest2, testfile), axis=0)

    np.random.shuffle(alltrain)
    np.random.shuffle(alltrain2)
    #np.random.shuffle(alltest)
    #np.random.shuffle(alltest2)
    trainfile = np.concatenate((alltrain, alltrain2), axis=0)
    #np.random.shuffle(trainfile)
    print(trainfile.shape)

    testfile = np.concatenate((alltest, alltest2), axis=0)
    print(testfile.shape)


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
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
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
    trainfile = trainfile[:15]
    np.random.shuffle(trainfile)

    for j in ["1M"]:  # "1S", "2S"
        for i in [i for i in range(0, 20)]:
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
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
    trainfile2 = trainfile2[:15]
    np.random.shuffle(trainfile2)

    testfile = trainfile[10:]
    trainfile = trainfile[:15]
    testfile2 = trainfile2[10:]
    trainfile2 = trainfile2[:15]

    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile
img_rows = 9
img_cols = 10
channels = 1
img_shape = (img_rows, img_cols, channels)

epochs = 5000
batch_size = 20000
sample_interval = 100

train_feature, test_feature,train_feature_ot,train_domain_label,train_label,test_label,train_label_ot=read_datamid_huodong()
print("train_feature"+str(train_feature.shape))
print("test_feature"+str(test_feature.shape))
print("train_feature_ot"+str(train_feature_ot.shape))
print("train_domain_label"+str(train_domain_label.shape))
print("train_label"+str(train_label.shape))
print("test_label"+str(test_label.shape))
print("train_label_ot"+str(train_label_ot.shape))

#全局归化为0~1
a=np.concatenate((train_feature, test_feature), axis=0)
train_feature = (train_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
test_feature = (test_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
train_feature_ot=(train_feature_ot.astype('float32')-np.min(a))/(np.max(a)-np.min(a))

train_feature= train_feature.reshape([train_feature.shape[0], img_rows, img_cols])
test_feature= test_feature.reshape([test_feature.shape[0], img_rows, img_cols])
train_feature_ot= train_feature_ot.reshape([train_feature_ot.shape[0], img_rows, img_cols])
train_feature = np.expand_dims(train_feature, axis=3)
test_feature = np.expand_dims(test_feature, axis=3)
train_feature_ot = np.expand_dims(train_feature_ot, axis=3)


latent_dim = 90
latent_dim2=180

def build_ed(latent_dim2, img_shape):
    deterministic = 1
    img = Input(shape=img_shape)
    h = Flatten()(img)
    h = Dense(400, activation="relu")(h)
    h = Dense(400, activation="relu")(h)
    h = Dense(400, activation="relu")(h)
    latent_repr = Dense(latent_dim2)(h)
    return Model(img, latent_repr)
def build_class(latent_dim):
    model = Sequential()
    model.add(Dense(400, input_dim=latent_dim, activation="relu"))
    model.add(Dense(400, activation="relu"))
    model.add(Dense(400, activation="relu"))
    model.add(Dense(6, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_lstm():
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(nb_time_steps, nb_input_vector)))
    model.add(Dense(6, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_dis(latent_dim):
    model = Sequential()
    model.add(Dense(400, input_dim=latent_dim, activation="relu"))
    model.add(Dense(400, activation="relu"))
    model.add(Dense(400, activation="relu"))
    model.add(Dense(9, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_dd(latent_dim2, img_shape):
    model = Sequential()
    model.add(Dense(400, input_dim=latent_dim2,activation="relu"))
    model.add(Dense(400, activation="relu"))
    model.add(Dense(400, activation="relu"))
    model.add(Dense(np.prod(img_shape), activation='sigmoid'))
    model.add(Reshape(img_shape))
    z = Input(shape=(latent_dim2,))
    img = model(z)
    return Model(z, img)

opt = Adam(0.0002, 0.5)
classer = build_class(latent_dim)
classer.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
dis = build_dis(latent_dim)
dis.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
ed = build_ed(latent_dim2, img_shape)
dd = build_dd(latent_dim2, img_shape)

img3 = Input(shape=img_shape)
encoded_repr3 = ed(img3)
reconstructed_img3 = dd(encoded_repr3)
sc_fido = Model(img3,reconstructed_img3)
sc_fido.compile(loss='mse', optimizer=opt)
def get_class(x):
    return x[:,:latent_dim]
def get_dis(x):
    return x[:,latent_dim:]
encoded_repr3_class = Lambda(get_class)(encoded_repr3)
encoded_repr3_dis = Lambda(get_dis)(encoded_repr3)
validity1 = classer(encoded_repr3_class)
validity2 = dis(encoded_repr3_dis)
class_model=Model(img3,validity1)
class_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
dis_model=Model(img3,validity2)
dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

k=0
for epoch in range(epochs):

    # ---------------------
    #  Train classer
    # ---------------------

    # Select a random batch of images
    idx2 = np.random.randint(0, train_feature.shape[0], batch_size)
    imgs2 = train_feature[idx2]
    sc_fido_loss = sc_fido.train_on_batch(imgs2, imgs2)
    idx = np.random.randint(0, train_feature.shape[0], batch_size)
    imgs = train_feature[idx]
    d_loss = dis_model.train_on_batch(imgs, train_domain_label[idx])
    c_loss = class_model.train_on_batch(imgs, train_label[idx])

    if epoch % 10 == 0:
        print("%d [活动分类loss: %f,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,重构loss: %f]" % (
            epoch, c_loss[0], 100 * c_loss[1], d_loss[0], 100 * d_loss[1], sc_fido_loss))
        n = 0
        n1=0
        for k in range(9):
            non_mid = ed.predict(test_feature[k*5*6*200:(k+1)*5*6*200])
            non_mid = non_mid[:, :latent_dim]
            non_pre = classer.predict(non_mid)
            m = 0
            for i in range(6):
                for k in range(5*200):
                    x = np.argmax(non_pre[i * 5*200 + k])
                    if (x == i):
                        m = m + 1
                        n=n+1
            acc = float(m) / float(len(non_pre))
            print("源"+str(k+1)+"测试数据准确率：" + str(acc), end='   ')

            m1 = 0
            for i in range(6):
                for k in range(5):
                    y = 0
                    for j in range(200):
                        x = np.argmax(non_pre[(i * 5 * 200) + k * 200 + j])
                        if (x == i):
                            y = y + 1
                    if (y >= 100):
                        m1 = m1+1
                        n1=n1+1
            acc1 = float(m1) / float(30)
            print("投票后：" + str(acc))

        ac = float(n) / float(len(test_feature))
        ac1 = float(n1) / float(270)
        k1=ac
        k2=ac1
        print("源平均测试数据准确率：" + str(ac), end='   ')
        print("投票后：" + str(ac1))
        print()

        n = 0
        n1 = 0
        non_mid = ed.predict(train_feature_ot)
        non_mid = non_mid[:, :latent_dim]
        non_pre = classer.predict(non_mid)
        m = 0
        for i in range(6):
            for k in range(20 * 200):
                x = np.argmax(non_pre[i * 20 * 200 + k])
                if (x == i):
                    m = m + 1
                    n = n + 1
        acc = float(m) / float(len(non_pre))
        print("目标测试数据准确率：" + str(acc), end='   ')

        m1 = 0
        for i in range(6):
            for k in range(20):
                y = 0
                for j in range(200):
                    x = np.argmax(non_pre[(i * 20 * 200) + k * 200 + j])
                    if (x == i):
                        y = y + 1
                if (y >= 100):
                    m1 = m1 + 1
                    n1 = n1 + 1
        acc1 = float(m1) / float(20*6)
        kk1 = acc
        kk2 = acc1
        print("投票后：" + str(acc))
        print()


        # if ((acc_non_pre3_vot >= 0.66) and (acc_yes_pre3_vot >= 0.66) and (c_loss[1] >= 0.65) and (
        #         acc_non_pre4_vot >= 0.66) and (acc_yes_pre4_vot >= 0.66)):
        if ((k2 >= 0.65) and(kk2 >= 0.65)):
            k1 = k1 * 1000
            k1 = int(k1)
            k2 = k2 * 1000
            k2 = int(k2)

            kk1 = kk1 * 1000
            kk1 = int(kk1)
            kk2 = kk2 * 1000
            kk2 = int(kk2)
            c = 100 * c_loss[1]
            c = int(c)
            file = r'models/to-1/result_dingwei.txt'
            f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
            str1 = str(epoch) + 'mid_' + str(c) + 'y_' + str(
                k1) + '_' + str(k2) + 'm' + str(kk1) + '_' + str(
                kk2) + '\n'
            f.write(str1.encode())
            f.close()  # 关闭文件
    if epoch == 500:
        classer.save_weights('models/to-1/500classer.h5')
        ed.save_weights('models/to-1/500ed.h5')
        dd.save_weights('models/to-1/500dd.h5')
        dis.save_weights('models/to-1/500dis.h5')
    if epoch == 1000:
        classer.save_weights('models/to-1/1000classer.h5')
        ed.save_weights('models/to-1/1000ed.h5')
        dd.save_weights('models/to-1/1000dd.h5')
        dis.save_weights('models/to-1/1000dis.h5')

    if epoch == 2000:
        classer.save_weights('models/to-1/2000classer.h5')
        ed.save_weights('models/to-1/2000ed.h5')
        dd.save_weights('models/to-1/2000dd.h5')
        dis.save_weights('models/to-1/2000dis.h5')

    if epoch == 3000:
        classer.save_weights('models/to-1/3000classer.h5')
        ed.save_weights('models/to-1/3000ed.h5')
        dd.save_weights('models/to-1/3000dd.h5')
        dis.save_weights('models/to-1/3000dis.h5')

    if epoch == 4000:
        classer.save_weights('models/to-1/4000classer.h5')
        ed.save_weights('models/to-1/4000ed.h5')
        dd.save_weights('models/to-1/4000dd.h5')
        dis.save_weights('models/to-1/4000dis.h5')

classer.save_weights('models/to-1/classer.h5')
ed.save_weights('models/to-1/ed.h5')
dd.save_weights('models/to-1/dd.h5')
dis.save_weights('models/to-1/dis.h5')

localtime2 = time.asctime( time.localtime(time.time()) )
print ("开始时间为 :", localtime1)
print ("结束时间为 :", localtime2)