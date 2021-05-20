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
from keras.layers import Input, Dense, Reshape, Flatten, LSTM, multiply, GaussianNoise,Bidirectional
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
cut1=0
cut2_0=15
cut2_1M=15
lin=115
lincut=120
ww=1
lin2=int((lin*2)/ww)
lincut2=int((lincut*2)/ww)
linlong=162
nb_lstm_outputs = 50  #神经元个数
nb_time_steps = lin2-cut1*2  #时间序列长度
nb_input_vector = 270 #输入序列
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
    label = np_utils.to_categorical(label)
    label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label), np.array(label2)
def read_data_cut2(filenames):
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
        if ('-0-' in filename):
            idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lincut,a-cut2_0, ww)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(a+cut2_0,int(temp_feature.shape[0] / 2) + lincut, ww)])  # 取中心点处左右分布数据
            idx = np.hstack((idx1, idx2))
            temp_feature = temp_feature[idx]
        if ('-1M-' in filename):
            idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lincut,a-cut2_1M, ww)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(a+cut2_1M,int(temp_feature.shape[0] / 2) + lincut, ww)])  # 取中心点处左右分布数据
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
    label = np_utils.to_categorical(label)
    label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label),np.array(label2)
def read_data_lstm(filenames):
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
        temp_feature = np.expand_dims(temp_feature, axis=0)
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
        temp_label = np.tile(temp_label, (1,))
        temp_label2 = np.tile(temp_label2, (1,))
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
def read_data_cut_lstm(filenames):
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
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2) - lincut,
                                         int(csvdata.shape[0] / 2) + lincut, ww)])  # 取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature
        feat = np.sum(feat, axis=1)
        feat = np.rint(feat)
        a = np.argmax(feat)  # 返回feature最大值位置
        idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lincut, a - cut1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a + cut1, int(temp_feature.shape[0] / 2) + lincut, ww)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        temp_feature = temp_feature[idx]
        temp_feature = np.expand_dims(temp_feature, axis=0)
        # print(temp_feature)
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

        temp_label = np.tile(temp_label, (1,))
        temp_label2 = np.tile(temp_label2, (1,))
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
def read_data_cut_lstmmid(filenames):
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
                                         int(csvdata.shape[0] / 2) + linlong, ww)])  # 取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature
        feat = np.sum(feat, axis=1)
        feat = np.rint(feat)
        a = np.argmax(feat)  # 返回feature最大值位置
        idx1 = np.array([j for j in range(a - lin, a - cut1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a + cut1, a + lin, ww)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        temp_feature = temp_feature[idx]
        temp_feature = np.expand_dims(temp_feature, axis=0)
        # print(temp_feature)
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
        elif ('ljc' in filename):
            temp_label2 = 3
        elif ('cyh' in filename):
            temp_label2 = 4
        elif ('tk' in filename):
            temp_label2 = 5
        elif ('lyx' in filename):
            temp_label2 = 6

        temp_label = np.tile(temp_label, (1,))
        temp_label2 = np.tile(temp_label2, (1,))
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
def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/caiji/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['zb','zhw', 'gzy', 'ljc', 'cyh', 'tk']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]

    trainfile += filenames[:120]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable,domain_label = read_data(trainfile)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    print(feature)
    k = np.arange(120)
    for i in range(0, 120):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:115]
    #np.random.shuffle(trainfile)

    for name in ['zb','zhw', 'gzy', 'ljc', 'cyh', 'tk']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 20)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile2 += filenames[:120]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable,domain_label = read_data(trainfile2)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(120)
    for i in range(0, 120):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:115]
    #np.random.shuffle(trainfile2)

    testfile = trainfile[55:70]
    trainfile = np.concatenate((trainfile[:55], trainfile[70:]), axis=0)
    np.random.shuffle(trainfile)
    testfile2 = trainfile2[55:70]
    trainfile2 = np.concatenate((trainfile2[:55], trainfile2[70:]), axis=0)
    np.random.shuffle(trainfile2)

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
            fn = filepath + "lyx-2.5-M/" + "lyx-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile += filenames[:20]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable,domain_label = read_data(trainfile)

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
            fn = filepath + "lyx-2.5-M/" + "lyx-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile2 += filenames[:20]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable,domain_label = read_data(trainfile2)

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
# Results can be found in just_2_rv
# latent_dim = 2
latent_dim = 10



epochs = 5000
batch_size = 100
sample_interval = 100


# Rescale -1 to 1
trainfile_array, testfile_array = file_array()#

train_feature_cut, train_label_cut,train_domain_label_cut = read_data_cut_lstmmid(trainfile_array)
test_feature_cut, test_label_cut,test_domain_label_cut = read_data_cut_lstmmid(testfile_array)

trainfile_other, testfile_other = other_file_array()#
train_feature_ot_cut, train_label_ot_cut,train_domain_label_ot_cut = read_data_cut_lstmmid(trainfile_other)
#全局归化为0~1
#a=np.concatenate((train_feature, train_feature_ot), axis=0)
a=train_feature_cut
train_feature_cut = (train_feature_cut.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
test_feature_cut = (test_feature_cut.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
train_feature_ot_cut=(train_feature_ot_cut.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
X_train1 =train_feature_cut[:100]
print(X_train1.shape)
X_test1 =test_feature_cut[:15]
print(X_test1.shape)

X_train2 =train_feature_cut[100:]
print(X_train2.shape)
X_test2 =test_feature_cut[15:]
print(X_test2.shape)


print("train_feature_ot")
print(train_feature_ot_cut.shape)

def build_lstm():
    model = Sequential()
    model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
# def build_lstm():
#     model = Sequential()
#     model.add(Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), input_shape=(nb_time_steps, nb_input_vector)))
#     model.add(Dense(2, activation="softmax"))
#     encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
#     validity = model(encoded_repr)
#     return Model(encoded_repr, validity)

opt = Adam(0.0002, 0.5)
lstm = build_lstm()
lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
img3 = Input(shape=(nb_time_steps, nb_input_vector))
validity = lstm(img3)
lstm_model = Model(img3,validity)
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

k=0
for epoch in range(epochs):

    # ---------------------
    #  Train classer
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, train_feature_cut.shape[0], batch_size)
    imgs = train_feature_cut[idx]
    l_loss = lstm_model.train_on_batch(imgs, train_label_cut[idx])
    # ---------------------
    #  Train dis
    # ---------------------


    # ---------------------
    #  Train chonggou
    # ---------------------


    # Plot the progress (every 10th epoch)
    if epoch % 1 == 0:
        print("%d [危险品分类loss: %f,acc: %.2f%%]" % (epoch, l_loss[0], 100 * l_loss[1]))

        non_pre = lstm.predict(test_feature_cut[:15])
        yes_pre = lstm.predict(test_feature_cut[15:])

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
        a2 = [0, 0]
        k1 = [0, 0]
        for i in range(0, int(len(yes_pre))):
            if yes_pre[i][0] > yes_pre[i][1]: a1[0] = a1[0] + 1
            if yes_pre[i][0] <= yes_pre[i][1]: a1[1] = a1[1] + 1

        a1 = [0, 0]
        yes_pre_1 = np.arange(len(yes_pre))
        for i in range(0, int(len(yes_pre))):
            if yes_pre[i][0] > yes_pre[i][1]:
                a1[0] = a1[0] + 1
                yes_pre_1[i] = 1
            if yes_pre[i][0] <= yes_pre[i][1]:
                a1[1] = a1[1] + 1
                yes_pre_1[i] = 0

        acc_yes_pre = float(a1[1]) / float(len(yes_pre))

        print('源数切割前正确率：', end='   ')
        print(acc_non_pre, end='   ')
        print(acc_yes_pre, end='   ')
        print()

        non_pre3 = lstm.predict(train_feature_ot_cut[:15])
        yes_pre3 = lstm.predict(train_feature_ot_cut[15:])

        a1 = [0, 0]
        a2 = [0, 0]
        k1 = [0, 0]
        non_pre3_1 = np.arange(len(non_pre3))
        for i in range(0, int(len(non_pre3))):
            if non_pre3[i][0] >= non_pre3[i][1]:
                a1[0] = a1[0] + 1
                non_pre3_1[i] = 1
            if non_pre3[i][0] < non_pre3[i][1]:
                a1[1] = a1[1] + 1
                non_pre3_1[i] = 0

        acc_non_pre3 = float(a1[0]) / float(len(non_pre3))
        a1 = [0, 0]
        a2 = [0, 0]
        k1 = [0, 0]
        for i in range(0, int(len(yes_pre3))):
            if yes_pre3[i][0] > yes_pre3[i][1]: a1[0] = a1[0] + 1
            if yes_pre3[i][0] <= yes_pre3[i][1]: a1[1] = a1[1] + 1

        a1 = [0, 0]
        yes_pre3_1 = np.arange(len(yes_pre3))
        for i in range(0, int(len(yes_pre3))):
            if yes_pre3[i][0] > yes_pre3[i][1]:
                a1[0] = a1[0] + 1
                yes_pre3_1[i] = 1
            if yes_pre3[i][0] <= yes_pre3[i][1]:
                a1[1] = a1[1] + 1
                yes_pre3_1[i] = 0

        acc_yes_pre3 = float(a1[1]) / float(len(yes_pre3))

        print('目标域数据正确率：', end='   ')
        print(acc_non_pre3, end='   ')
        print(acc_yes_pre3, end='   ')
        print()
        if ((acc_non_pre >= 0.6) and (acc_yes_pre >= 0.6) and (l_loss[1] >= 0.6) and (
                acc_non_pre3 >= 0.6) and (acc_yes_pre3 >= 0.6)):
            k = k + 1
            acc_non_pre = acc_non_pre * 100
            acc_non_pre = int(acc_non_pre)
            acc_yes_pre = acc_yes_pre * 100
            acc_yes_pre = int(acc_yes_pre)

            acc_non_pre3 = acc_non_pre3 * 100
            acc_non_pre3 = int(acc_non_pre3)
            acc_yes_pre3 = acc_yes_pre3 * 100
            acc_yes_pre3 = int(acc_yes_pre3)


            c = 100 * l_loss[1]
            c = int(c)
            print(k)
            lstm.save_weights('models/lstm/' + str(epoch) + '_' + str(c) + 'y' + str(
                    acc_non_pre) + '_' + str(acc_yes_pre) + 'm' + str(acc_non_pre3) + '_' + str(acc_yes_pre3) + '_'  + 'classer.h5')

    if epoch == 500:
        lstm.save_weights('models/lstm/500lstm.h5')
    if epoch == 1000:
        lstm.save_weights('models/lstm/1000lstm.h5')
    if epoch == 2000:
        lstm.save_weights('models/lstm/2000lstm.h5')
    if epoch == 3000:
        lstm.save_weights('models/lstm/3000lstm.h5')
    if epoch == 4000:
        lstm.save_weights('models/lstm/4000lstm.h5')
print("%d [危险品分类loss: %f,acc: %.2f%%]" % (epoch, l_loss[0], 100 * l_loss[1]))
lstm.save_weights('models/lstm/lstm.h5')

localtime2 = time.asctime( time.localtime(time.time()) )
print ("开始时间为 :", localtime1)
print ("结束时间为 :", localtime2)