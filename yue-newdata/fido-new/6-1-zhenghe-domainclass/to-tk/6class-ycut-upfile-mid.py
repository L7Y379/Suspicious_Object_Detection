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
    return np.array(feature[:, :270]), np.array(label), np.array(label2)
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

    #np.random.shuffle(alltrain)
    #np.random.shuffle(alltrain2)
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
img_rows = 15
img_cols = 18
channels = 1
img_shape = (img_rows, img_cols, channels)

epochs = 5000
batch_size = 20000


# Rescale -1 to 1
trainfile_array, testfile_array = file_array3()#
print(trainfile_array)
print(testfile_array)
train_feature, train_label,train_domain_label = read_datamid(trainfile_array)
train_feature_cut, train_label_cut,train_domain_label_cut = read_data_cutmid(trainfile_array)
test_feature, test_label,test_domain_label = read_datamid(testfile_array)
test_feature_cut, test_label_cut,test_domain_label_cut = read_data_cutmid(testfile_array)

trainfile_other, testfile_other = other_file_arraymid()#
train_feature_ot, train_label_ot,train_domain_label_ot = read_datamid(trainfile_other)
train_feature_ot_cut, train_label_ot_cut,train_domain_label_ot_cut = read_data_cut2mid(trainfile_other)
print(train_feature_ot_cut)
test_feature_ot, test_label_ot,test_domain_label_ot = read_datamid(testfile_other)
#全局归化为0~1
#a=np.concatenate((train_feature, train_feature_ot), axis=0)
a=train_feature_cut
train_feature_cut = (train_feature_cut.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
test_feature_cut = (test_feature_cut.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
train_feature_ot=(train_feature_ot.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
train_feature_ot_cut=(train_feature_ot_cut.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
train_feature=(train_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
test_feature_ot=(test_feature_ot.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
test_feature=(test_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
X_train1 =train_feature_cut[:90*(lincut2 - cut1 * 2)]
print(X_train1.shape)
X_test1 =test_feature_cut[:15*(lincut2 - cut1 * 2)]
print(X_test1.shape)
X_train1 = X_train1.reshape([X_train1.shape[0], img_rows, img_cols])
X_test1 = X_test1.reshape([X_test1.shape[0], img_rows, img_cols])
X_train1 = np.expand_dims(X_train1, axis=3)
X_test1 = np.expand_dims(X_test1, axis=3)

X_train2 =train_feature_cut[90*(lincut2 - cut1 * 2):]
print(X_train2.shape)
X_test2 =test_feature_cut[15*(lincut2 - cut1 * 2):]
print(X_test2.shape)
X_train2 = X_train2.reshape([X_train2.shape[0], img_rows, img_cols])
X_test2 = X_test2.reshape([X_test2.shape[0], img_rows, img_cols])
X_train2 = np.expand_dims(X_train2, axis=3)
X_test2 = np.expand_dims(X_test2, axis=3)

train_feature_ot = train_feature_ot.reshape([train_feature_ot.shape[0], img_rows, img_cols])
train_feature_ot_cut = train_feature_ot_cut.reshape([train_feature_ot_cut.shape[0], img_rows, img_cols])
train_feature=train_feature.reshape([train_feature.shape[0], img_rows, img_cols])
test_feature = test_feature.reshape([test_feature.shape[0], img_rows, img_cols])
test_feature_ot = test_feature_ot.reshape([test_feature_ot.shape[0], img_rows, img_cols])
train_feature_ot = np.expand_dims(train_feature_ot, axis=3)
train_feature_ot_cut = np.expand_dims(train_feature_ot_cut, axis=3)
train_feature= np.expand_dims(train_feature, axis=3)
test_feature=np.expand_dims(test_feature, axis=3)
test_feature_ot = np.expand_dims(test_feature_ot, axis=3)
print("train_feature_ot")
print(train_feature_ot.shape)
print(train_feature_ot_cut.shape)

# X_SCdata1=0.5*X_train1+0.5*scdata1
# X_SCdata2=0.5*X_train2+0.5*scdata2
X_SCdata1=X_train1
X_SCdata2=X_train2
X_SCdata1_label=train_label_cut[:90*(lincut2 - cut1 * 2)]
X_SCdata2_label=train_label_cut[90*(lincut2 - cut1 * 2):]
X_SCdata1_domain_label=train_domain_label_cut[:90*(lincut2 - cut1 * 2)]
X_SCdata2_domain_label=train_domain_label_cut[90*(lincut2 - cut1 * 2):]

X_SCdata=np.concatenate((X_train1,X_train2), axis=0)
# X_SCdata=np.concatenate((X_SCdata,X_SCdata1), axis=0)
# X_SCdata=np.concatenate((X_SCdata,X_SCdata2), axis=0)
X_SCdata_label=np.concatenate((X_SCdata1_label,X_SCdata2_label), axis=0)
#X_SCdata_label=np.concatenate((X_SCdata_label,X_SCdata_label), axis=0)
X_SCdata_domain_label=np.concatenate((X_SCdata1_domain_label,X_SCdata2_domain_label), axis=0)
#X_SCdata_domain_label=np.concatenate((X_SCdata_domain_label,X_SCdata_domain_label), axis=0)

all_data=X_SCdata
print(all_data.shape)
latent_dim = 270
latent_dim2=540

X_SCdata1=np.concatenate((X_SCdata[:15*(lincut2 - cut1 * 2)],X_SCdata[90*(lincut2 - cut1 * 2):105*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata2=np.concatenate((X_SCdata[15*(lincut2 - cut1 * 2):30*(lincut2 - cut1 * 2)],X_SCdata[105*(lincut2 - cut1 * 2):120*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata3=np.concatenate((X_SCdata[30*(lincut2 - cut1 * 2):45*(lincut2 - cut1 * 2)],X_SCdata[120*(lincut2 - cut1 * 2):135*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata4=np.concatenate((X_SCdata[45*(lincut2 - cut1 * 2):60*(lincut2 - cut1 * 2)],X_SCdata[135*(lincut2 - cut1 * 2):150*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata5=np.concatenate((X_SCdata[60*(lincut2 - cut1 * 2):75*(lincut2 - cut1 * 2)],X_SCdata[150*(lincut2 - cut1 * 2):165*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata6=np.concatenate((X_SCdata[75*(lincut2 - cut1 * 2):90*(lincut2 - cut1 * 2)],X_SCdata[165*(lincut2 - cut1 * 2):180*(lincut2 - cut1 * 2)]), axis=0)

X_SCdata_label1=np.concatenate((X_SCdata_label[:15*(lincut2 - cut1 * 2)],X_SCdata_label[90*(lincut2 - cut1 * 2):105*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata_label2=np.concatenate((X_SCdata_label[15*(lincut2 - cut1 * 2):30*(lincut2 - cut1 * 2)],X_SCdata_label[105*(lincut2 - cut1 * 2):120*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata_label3=np.concatenate((X_SCdata_label[30*(lincut2 - cut1 * 2):45*(lincut2 - cut1 * 2)],X_SCdata_label[120*(lincut2 - cut1 * 2):135*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata_label4=np.concatenate((X_SCdata_label[45*(lincut2 - cut1 * 2):60*(lincut2 - cut1 * 2)],X_SCdata_label[135*(lincut2 - cut1 * 2):150*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata_label5=np.concatenate((X_SCdata_label[60*(lincut2 - cut1 * 2):75*(lincut2 - cut1 * 2)],X_SCdata_label[150*(lincut2 - cut1 * 2):165*(lincut2 - cut1 * 2)]), axis=0)
X_SCdata_label6=np.concatenate((X_SCdata_label[75*(lincut2 - cut1 * 2):90*(lincut2 - cut1 * 2)],X_SCdata_label[165*(lincut2 - cut1 * 2):180*(lincut2 - cut1 * 2)]), axis=0)
def build_ed(latent_dim2, img_shape):
    deterministic = 1
    img = Input(shape=img_shape)
    h = Flatten()(img)
    h = Dense(800, activation="relu")(h)
    h = Dense(800, activation="relu")(h)
    h = Dense(800, activation="relu")(h)
    latent_repr = Dense(latent_dim2)(h)
    return Model(img, latent_repr)
def build_class1(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_class2(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_class3(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_class4(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_class5(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_class6(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_dis(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(6, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_dd(latent_dim2, img_shape):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim2,activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(np.prod(img_shape), activation='sigmoid'))
    model.add(Reshape(img_shape))
    z = Input(shape=(latent_dim2,))
    img = model(z)
    return Model(z, img)

opt = Adam(0.0002, 0.5)
classer1 = build_class1(latent_dim)
classer1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
classer2 = build_class2(latent_dim)
classer2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
classer3 = build_class3(latent_dim)
classer3.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
classer4 = build_class4(latent_dim)
classer4.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
classer5 = build_class5(latent_dim)
classer5.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
classer6 = build_class6(latent_dim)
classer6.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
validity1 = classer1(encoded_repr3_class)
validity2 = dis(encoded_repr3_dis)
dis_model=Model(img3,validity2)
dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# # Training
# classer.load_weights('models/6classer/4000classer.h5')
ed.load_weights('models/6classer/ed.h5')
dd.load_weights('models/6classer/dd.h5')
dis.load_weights('models/6classer/dis.h5')
# dis_model.load_weights('models/6classer/1000dis_model.h5')
# class_model.load_weights('models/6classer/1000class_model.h5')
# sc_fido.load_weights('models/6classer/1000sc_fido.h5')


k=0
for epoch in range(epochs):

    # ---------------------
    #  Train classer
    # ---------------------

    # Select a random batch of images
    idx2 = np.random.randint(0, all_data.shape[0], batch_size)
    imgs2 = all_data[idx2]
    sc_fido_loss = sc_fido.train_on_batch(imgs2, imgs2)
    idx = np.random.randint(0, X_SCdata.shape[0], batch_size)
    imgs = all_data[idx]
    d_loss = dis_model.train_on_batch(imgs, X_SCdata_domain_label[idx])

    idx3 = np.random.randint(0, X_SCdata1.shape[0], int(batch_size / 6))
    imgs1 = X_SCdata1[idx3]
    imgs2 = X_SCdata2[idx3]
    imgs3 = X_SCdata3[idx3]
    imgs4 = X_SCdata4[idx3]
    imgs5 = X_SCdata5[idx3]
    imgs6 = X_SCdata6[idx3]
    latent_mid1 = ed.predict(imgs1)
    latent_mid1 = latent_mid1[:, :latent_dim]
    latent_mid2 = ed.predict(imgs2)
    latent_mid2 = latent_mid2[:, :latent_dim]
    latent_mid3 = ed.predict(imgs3)
    latent_mid3 = latent_mid3[:, :latent_dim]
    latent_mid4 = ed.predict(imgs4)
    latent_mid4 = latent_mid4[:, :latent_dim]
    latent_mid5 = ed.predict(imgs5)
    latent_mid5 = latent_mid5[:, :latent_dim]
    latent_mid6 = ed.predict(imgs6)
    latent_mid6 = latent_mid6[:, :latent_dim]
    c_loss1 = classer1.train_on_batch(latent_mid1, X_SCdata_label1[idx3])
    c_loss2 = classer2.train_on_batch(latent_mid2, X_SCdata_label2[idx3])
    c_loss3 = classer3.train_on_batch(latent_mid3, X_SCdata_label3[idx3])
    c_loss4 = classer4.train_on_batch(latent_mid4, X_SCdata_label4[idx3])
    c_loss5 = classer5.train_on_batch(latent_mid5, X_SCdata_label5[idx3])
    c_loss6 = classer6.train_on_batch(latent_mid6, X_SCdata_label6[idx3])

    if epoch % 10 == 0:
        print("%d [危险品分类acc: %.2f%%,acc: %.2f%%,acc: %.2f%%,acc: %.2f%%,acc: %.2f%%,acc: %.2f%%域分类loss: %f,acc: %.2f%%,重构loss: %f]" % (
        epoch,100 * c_loss1[1],100 * c_loss2[1],100 * c_loss3[1],100 * c_loss4[1],100 * c_loss5[1],100 * c_loss6[1],d_loss[0],100 * d_loss[1], sc_fido_loss))


        non_mid = ed.predict(X_test1)
        non_mid = non_mid[:, :latent_dim]
        non_pre1 = classer1.predict(non_mid[:1*(lincut2 - cut1 * 2)])
        non_pre2 = classer2.predict(non_mid[1*(lincut2 - cut1 * 2):5*(lincut2 - cut1 * 2)])
        non_pre3 = classer3.predict(non_mid[5*(lincut2 - cut1 * 2):7*(lincut2 - cut1 * 2)])
        non_pre4 = classer4.predict(non_mid[7*(lincut2 - cut1 * 2):9*(lincut2 - cut1 * 2)])
        non_pre5 = classer5.predict(non_mid[9*(lincut2 - cut1 * 2):12*(lincut2 - cut1 * 2)])
        non_pre6 = classer6.predict(non_mid[12*(lincut2 - cut1 * 2):15*(lincut2 - cut1 * 2)])
        yes_mid = ed.predict(X_test2)
        yes_mid = yes_mid[:, :latent_dim]
        yes_pre1 = classer1.predict(yes_mid[:1 * (lincut2 - cut1 * 2)])
        yes_pre2 = classer2.predict(yes_mid[1 * (lincut2 - cut1 * 2):5 * (lincut2 - cut1 * 2)])
        yes_pre3 = classer3.predict(yes_mid[5 * (lincut2 - cut1 * 2):7 * (lincut2 - cut1 * 2)])
        yes_pre4 = classer4.predict(yes_mid[7 * (lincut2 - cut1 * 2):9 * (lincut2 - cut1 * 2)])
        yes_pre5 = classer5.predict(yes_mid[9 * (lincut2 - cut1 * 2):12 * (lincut2 - cut1 * 2)])
        yes_pre6 = classer6.predict(yes_mid[12 * (lincut2 - cut1 * 2):15 * (lincut2 - cut1 * 2)])
        kk = 0
        kk2 = 0
        kkk=0
        kkk2=0
        for y in range(6):
            if y==0:
                non_pre=non_pre1
                yes_pre=yes_pre1
            if y==1:
                non_pre=non_pre2
                yes_pre=yes_pre2
            if y==2:
                non_pre=non_pre3
                yes_pre=yes_pre3
            if y==3:
                non_pre=non_pre4
                yes_pre=yes_pre4
            if y==4:
                non_pre=non_pre5
                yes_pre=yes_pre5
            if y==5:
                non_pre=non_pre6
                yes_pre=yes_pre6
            a1 = [0, 0]
            a2 = [0, 0]
            k1 = [0, 0]
            non_pre_1 = np.arange(len(non_pre))
            for i in range(0, int(len(non_pre))):
                if non_pre[i][0] >= non_pre[i][1]:
                    a1[0] = a1[0] + 1
                    non_pre_1[i] = 1
                if non_pre[i][0] < non_pre[i][1]:
                    a1[1] = a1[1] + 1
                    non_pre_1[i] = 0
            acc_non_pre = float(a1[0]) / float(len(non_pre))
            kk=kk+acc_non_pre
            a1 = [0, 0]
            for i in range(0, int(len(non_pre_1))):
                if non_pre_1[i] == 1:
                    k1[0] = k1[0] + 1
                if non_pre_1[i] == 0:
                    k1[1] = k1[1] + 1
                if (k1[0] + k1[1] == (lincut2 - cut1 * 2)):
                    if k1[0] >= k1[1]:
                        a2[0] = a2[0] + 1
                    if k1[0] < k1[1]:
                        a2[1] = a2[1] + 1
                    k1 = [0, 0]
            acc_non_pre_vot = float(a2[0]) / float(len(non_pre_1) / (lincut2 - cut1 * 2))
            kk2=kk2+acc_non_pre_vot
            a1 = [0, 0]
            a2 = [0, 0]
            k1 = [0, 0]
            yes_pre_1 = np.arange(len(yes_pre))
            for i in range(0, int(len(yes_pre))):
                if yes_pre[i][0] > yes_pre[i][1]:
                    a1[0] = a1[0] + 1
                    yes_pre_1[i] = 1
                if yes_pre[i][0] <= yes_pre[i][1]:
                    a1[1] = a1[1] + 1
                    yes_pre_1[i] = 0
            acc_yes_pre = float(a1[1]) / float(len(yes_pre))
            kkk=acc_yes_pre+kkk
            a1 = [0, 0]
            for i in range(0, int(len(yes_pre_1))):
                if yes_pre_1[i] == 1:
                    k1[0] = k1[0] + 1
                if yes_pre_1[i] == 0:
                    k1[1] = k1[1] + 1
                if (k1[0] + k1[1] == (lincut2 - cut1 * 2)):
                    if k1[0] > k1[1]:
                        a2[0] = a2[0] + 1
                    if k1[0] <= k1[1]:
                        a2[1] = a2[1] + 1
                    k1 = [0, 0]
            acc_yes_pre_vot = float(a2[1]) / float(len(yes_pre_1) / (lincut2 - cut1 * 2))
            kkk2 = acc_yes_pre_vot + kkk2
            print('源数'+str(y+1)+'切割后正确率：', end='   ')
            print(acc_non_pre, end='   ')
            print(acc_yes_pre, end='   ')
            print(acc_non_pre_vot, end='   ')
            print(acc_yes_pre_vot)
        acc_non_pre=float(kk) / float(6)
        acc_non_pre_vot = float(kk2) / float(6)
        acc_yes_pre = float(kkk) / float(6)
        acc_yes_pre_vot = float(kkk2) / float(6)

        print('源数平均正确率：', end='   ')
        print(acc_non_pre, end='   ')
        print(acc_yes_pre, end='   ')
        print(acc_non_pre_vot, end='   ')
        print(acc_yes_pre_vot)
        kk = acc_non_pre
        kk2= acc_non_pre_vot
        kkk= acc_yes_pre
        kkk2= acc_yes_pre_vot
        non_mid4 = ed.predict(train_feature_ot_cut[:(lincut2 - cut2_0 * 2) * 15])
        non_mid4 = non_mid4[:, :latent_dim]
        non_pre1 = classer1.predict(non_mid4)
        non_pre2 = classer2.predict(non_mid4)
        non_pre3 = classer3.predict(non_mid4)
        non_pre4 = classer4.predict(non_mid4)
        non_pre5 = classer5.predict(non_mid4)
        non_pre6 = classer6.predict(non_mid4)
        yes_mid4 = ed.predict(train_feature_ot_cut[(lincut2 - cut2_0 * 2) * 15:])
        yes_mid4 = yes_mid4[:, :latent_dim]
        yes_pre1 = classer1.predict(yes_mid4)
        yes_pre2 = classer2.predict(yes_mid4)
        yes_pre3 = classer3.predict(yes_mid4)
        yes_pre4 = classer4.predict(yes_mid4)
        yes_pre5 = classer5.predict(yes_mid4)
        yes_pre6 = classer6.predict(yes_mid4)

        for y in range(6):
            if y==0:
                non_pre=non_pre1
                yes_pre=yes_pre1
            if y==1:
                non_pre=non_pre2
                yes_pre=yes_pre2
            if y==2:
                non_pre=non_pre3
                yes_pre=yes_pre3
            if y==3:
                non_pre=non_pre4
                yes_pre=yes_pre4
            if y==4:
                non_pre=non_pre5
                yes_pre=yes_pre5
            if y==5:
                non_pre=non_pre6
                yes_pre=yes_pre6
            a1 = [0, 0]
            a2 = [0, 0]
            k1 = [0, 0]
            non_pre_1 = np.arange(len(non_pre))
            for i in range(0, int(len(non_pre))):
                if non_pre[i][0] >= non_pre[i][1]:
                    a1[0] = a1[0] + 1
                    non_pre_1[i] = 1
                if non_pre[i][0] < non_pre[i][1]:
                    a1[1] = a1[1] + 1
                    non_pre_1[i] = 0
            acc_non_pre = float(a1[0]) / float(len(non_pre))
            a1 = [0, 0]
            for i in range(0, int(len(non_pre_1))):
                if non_pre_1[i] == 1:
                    k1[0] = k1[0] + 1
                if non_pre_1[i] == 0:
                    k1[1] = k1[1] + 1
                if (k1[0] + k1[1] == (lincut2 - cut1 * 2)):
                    if k1[0] >= k1[1]:
                        a2[0] = a2[0] + 1
                    if k1[0] < k1[1]:
                        a2[1] = a2[1] + 1
                    k1 = [0, 0]
            acc_non_pre_vot = float(a2[0]) / float(len(non_pre_1) / (lincut2 - cut1 * 2))
            a1 = [0, 0]
            a2 = [0, 0]
            k1 = [0, 0]
            yes_pre_1 = np.arange(len(yes_pre))
            for i in range(0, int(len(yes_pre))):
                if yes_pre[i][0] > yes_pre[i][1]:
                    a1[0] = a1[0] + 1
                    yes_pre_1[i] = 1
                if yes_pre[i][0] <= yes_pre[i][1]:
                    a1[1] = a1[1] + 1
                    yes_pre_1[i] = 0
            acc_yes_pre = float(a1[1]) / float(len(yes_pre))
            a1 = [0, 0]
            for i in range(0, int(len(yes_pre_1))):
                if yes_pre_1[i] == 1:
                    k1[0] = k1[0] + 1
                if yes_pre_1[i] == 0:
                    k1[1] = k1[1] + 1
                if (k1[0] + k1[1] == (lincut2 - cut1 * 2)):
                    if k1[0] > k1[1]:
                        a2[0] = a2[0] + 1
                    if k1[0] <= k1[1]:
                        a2[1] = a2[1] + 1
                    k1 = [0, 0]
            acc_yes_pre_vot = float(a2[1]) / float(len(yes_pre_1) / (lincut2 - cut1 * 2))
            print('目标'+str(y+1)+'切割后正确率：', end='   ')
            print(acc_non_pre, end='   ')
            print(acc_yes_pre, end='   ')
            print(acc_non_pre_vot, end='   ')
            print(acc_yes_pre_vot)

        dis_mid = ed.predict(train_feature_ot_cut[:(lincut2 - cut2_0 * 2) * 15])
        dis_mid = dis_mid[:, latent_dim:]
        dis_pre = dis.predict(dis_mid)
        dis_pre1 = dis_pre[:, :1]
        dis_pre2 = dis_pre[:, 1:2]
        dis_pre3 = dis_pre[:, 2:3]
        dis_pre4 = dis_pre[:, 3:4]
        dis_pre5 = dis_pre[:, 4:5]
        dis_pre6 = dis_pre[:, 5:6]
        print(dis_pre)
        non_pre = non_pre1 * dis_pre1+non_pre2 * dis_pre2+non_pre3 * dis_pre3+non_pre4 * dis_pre4+non_pre5 * dis_pre5+non_pre6 * dis_pre6
        dis_mid = ed.predict(train_feature_ot_cut[(lincut2 - cut2_0 * 2) * 15:])
        dis_mid = dis_mid[:, latent_dim:]
        dis_pre = dis.predict(dis_mid)
        dis_pre1 = dis_pre[:, :1]
        dis_pre2 = dis_pre[:, 1:2]
        dis_pre3 = dis_pre[:, 2:3]
        dis_pre4 = dis_pre[:, 3:4]
        dis_pre5 = dis_pre[:, 4:5]
        dis_pre6 = dis_pre[:, 5:6]
        print(dis_pre)
        yes_pre = yes_pre1 * dis_pre1 + yes_pre2 * dis_pre2 + yes_pre3 * dis_pre3 + yes_pre4 * dis_pre4 + yes_pre5 * dis_pre5 + yes_pre6 * dis_pre6

        a1 = [0, 0]
        a2 = [0, 0]
        k1 = [0, 0]
        non_pre4_1 = np.arange(len(non_pre))
        for i in range(0, int(len(non_pre))):
            if non_pre[i][0] >= non_pre[i][1]:
                a1[0] = a1[0] + 1
                non_pre4_1[i] = 1
            if non_pre[i][0] < non_pre[i][1]:
                a1[1] = a1[1] + 1
                non_pre4_1[i] = 0

        acc_non_pre4 = float(a1[0]) / float(len(non_pre))
        a1 = [0, 0]
        for i in range(0, int(len(non_pre4_1))):
            if non_pre4_1[i] == 1:
                k1[0] = k1[0] + 1

            if non_pre4_1[i] == 0:
                k1[1] = k1[1] + 1

            if (k1[0] + k1[1] == (lincut2 - cut2_0 * 2)):
                if k1[0] >= k1[1]:
                    a2[0] = a2[0] + 1
                if k1[0] < k1[1]:
                    a2[1] = a2[1] + 1
                k1 = [0, 0]
        acc_non_pre4_vot = float(a2[0]) / float(len(non_pre4_1) / (lincut2 - cut2_0 * 2))
        a1 = [0, 0]
        a2 = [0, 0]
        k1 = [0, 0]
        for i in range(0, int(len(yes_pre4))):
            if yes_pre4[i][0] > yes_pre4[i][1]: a1[0] = a1[0] + 1
            if yes_pre4[i][0] <= yes_pre4[i][1]: a1[1] = a1[1] + 1

        a1 = [0, 0]
        yes_pre4_1 = np.arange(len(yes_pre))
        for i in range(0, int(len(yes_pre))):
            if yes_pre[i][0] > yes_pre[i][1]:
                a1[0] = a1[0] + 1
                yes_pre4_1[i] = 1
            if yes_pre[i][0] <= yes_pre[i][1]:
                a1[1] = a1[1] + 1
                yes_pre4_1[i] = 0

        acc_yes_pre4 = float(a1[1]) / float(len(yes_pre))
        a1 = [0, 0]
        for i in range(0, int(len(yes_pre4_1))):
            if yes_pre4_1[i] == 1:
                k1[0] = k1[0] + 1

            if yes_pre4_1[i] == 0:
                k1[1] = k1[1] + 1

            if (k1[0] + k1[1] == (lincut2 - cut2_1M * 2)):
                if k1[0] > k1[1]:
                    a2[0] = a2[0] + 1
                if k1[0] <= k1[1]:
                    a2[1] = a2[1] + 1
                k1 = [0, 0]
        acc_yes_pre4_vot = float(a2[1]) / float(len(yes_pre4_1) / (lincut2 - cut2_1M* 2))
        print('数据正确率：', end='   ')
        print(acc_non_pre4, end='   ')
        print(acc_yes_pre4, end='   ')
        print(acc_non_pre4_vot, end='   ')
        print(acc_yes_pre4_vot)
        print()
        # if ((acc_non_pre3_vot >= 0.66) and (acc_yes_pre3_vot >= 0.66) and (c_loss[1] >= 0.65) and (
        #         acc_non_pre4_vot >= 0.66) and (acc_yes_pre4_vot >= 0.66)):
        if ((acc_non_pre4_vot >= 0.1) and (acc_yes_pre4_vot >= 0.1)):
            k = k + 1
            kk = kk * 100
            kk = int(kk)
            kkk = kkk * 100
            kkk = int(kkk)
            kk2 = kk2 * 100
            kk2 = int(kk2)
            kkk2 = kkk2 * 100
            kkk2 = int(kkk2)

            acc_non_pre4 = acc_non_pre4 * 100
            acc_non_pre4 = int(acc_non_pre4)
            acc_yes_pre4 = acc_yes_pre4 * 100
            acc_yes_pre4 = int(acc_yes_pre4)
            acc_non_pre4_vot = acc_non_pre4_vot * 100
            acc_non_pre4_vot = int(acc_non_pre4_vot)
            acc_yes_pre4_vot = acc_yes_pre4_vot * 100
            acc_yes_pre4_vot = int(acc_yes_pre4_vot)

            d = 100 * d_loss[1]
            d = int(d)
            file = r'D:\my bad\Suspicious object detection\Suspicious_Object_Detection\yue\yue-newdata\fido-new\6-1-zhenghe-domainclass\to-tk\models\6classer\test.txt'
            f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
            str1 = str(epoch) + 'mid_' +str(d) + 'y' + str(
                kk) + '_' + str(kkk) + '_' + str(kk2) + '_' + str(
                kkk2) + 'm' + str(acc_non_pre4) + '_' + str(
                acc_yes_pre4) + '_' + str(acc_non_pre4_vot) + '_' + str(acc_yes_pre4_vot)+'\n'
            f.write(str1.encode())
            f.close()  # 关闭文件
            print(k)
            # classer1.save_weights(
            #     'models/6classer/' + str(epoch) + 'mid_' +'y' + str(
            #         acc_non_pre) + '_' + str(acc_yes_pre) + '_' + str(acc_non_pre_vot) + '_' + str(
            #         acc_yes_pre_vot) + 'm' + str(acc_non_pre4) + '_' + str(
            #         acc_yes_pre4) + '_' + str(acc_non_pre4_vot) + '_' + str(acc_yes_pre4_vot) + 'classer1.h5')
            # ed.save_weights(
            #     'models/6classer/' + str(epoch) + 'mid_' +'y' + str(
            #         acc_non_pre) + '_' + str(acc_yes_pre) + '_' + str(acc_non_pre_vot) + '_' + str(
            #         acc_yes_pre_vot) + 'm' + str(acc_non_pre4) + '_' + str(
            #         acc_yes_pre4) + '_' + str(acc_non_pre4_vot) + '_' + str(acc_yes_pre4_vot) + 'ed.h5')

    if epoch == 500:
        classer1.save_weights('models/6classer/500classer1.h5')
        classer2.save_weights('models/6classer/500classer2.h5')
        classer3.save_weights('models/6classer/500classer3.h5')
        classer4.save_weights('models/6classer/500classer4.h5')
        classer5.save_weights('models/6classer/500classer5.h5')
        classer6.save_weights('models/6classer/500classer6.h5')
        ed.save_weights('models/6classer/500ed.h5')
        dd.save_weights('models/6classer/500dd.h5')
        dis.save_weights('models/6classer/500dis.h5')
    if epoch == 1000:
        classer1.save_weights('models/6classer/1000classer1.h5')
        classer2.save_weights('models/6classer/1000classer2.h5')
        classer3.save_weights('models/6classer/1000classer3.h5')
        classer4.save_weights('models/6classer/1000classer4.h5')
        classer5.save_weights('models/6classer/1000classer5.h5')
        classer6.save_weights('models/6classer/1000classer6.h5')
        ed.save_weights('models/6classer/1000ed.h5')
        dd.save_weights('models/6classer/1000dd.h5')
        dis.save_weights('models/6classer/1000dis.h5')
    if epoch == 2000:
        classer1.save_weights('models/6classer/2000classer1.h5')
        classer2.save_weights('models/6classer/2000classer2.h5')
        classer3.save_weights('models/6classer/2000classer3.h5')
        classer4.save_weights('models/6classer/2000classer4.h5')
        classer5.save_weights('models/6classer/2000classer5.h5')
        classer6.save_weights('models/6classer/2000classer6.h5')
        ed.save_weights('models/6classer/2000ed.h5')
        dd.save_weights('models/6classer/2000dd.h5')
        dis.save_weights('models/6classer/2000dis.h5')
    if epoch == 3000:
        classer1.save_weights('models/6classer/3000classer1.h5')
        classer2.save_weights('models/6classer/3000classer2.h5')
        classer3.save_weights('models/6classer/3000classer3.h5')
        classer4.save_weights('models/6classer/3000classer4.h5')
        classer5.save_weights('models/6classer/3000classer5.h5')
        classer6.save_weights('models/6classer/3000classer6.h5')
        ed.save_weights('models/6classer/3000ed.h5')
        dd.save_weights('models/6classer/3000dd.h5')
        dis.save_weights('models/6classer/3000dis.h5')
    if epoch == 4000:
        classer1.save_weights('models/6classer/4000classer1.h5')
        classer2.save_weights('models/6classer/4000classer2.h5')
        classer3.save_weights('models/6classer/4000classer3.h5')
        classer4.save_weights('models/6classer/4000classer4.h5')
        classer5.save_weights('models/6classer/4000classer5.h5')
        classer6.save_weights('models/6classer/4000classer6.h5')
        ed.save_weights('models/6classer/4000ed.h5')
        dd.save_weights('models/6classer/4000dd.h5')
        dis.save_weights('models/6classer/4000dis.h5')
classer1.save_weights('models/6classer/classer1.h5')
classer2.save_weights('models/6classer/classer2.h5')
classer3.save_weights('models/6classer/classer3.h5')
classer4.save_weights('models/6classer/classer4.h5')
classer5.save_weights('models/6classer/classer5.h5')
classer6.save_weights('models/6classer/classer6.h5')
ed.save_weights('models/6classer/ed.h5')
dd.save_weights('models/6classer/dd.h5')
dis.save_weights('models/6classer/dis.h5')

localtime2 = time.asctime( time.localtime(time.time()) )
print ("开始时间为 :", localtime1)
print ("结束时间为 :", localtime2)