#crnn dislstm
import pandas as pd
import os
from sklearn.cluster import KMeans
from keras.datasets import mnist
from keras.layers import Lambda
from keras import backend as K
from keras.layers import LSTM,Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D,TimeDistributed
from keras.layers import Lambda,Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
# from keras.optimizers import adam_v2
import numpy as np
from keras.utils import np_utils
import time
from sklearn.preprocessing import MinMaxScaler
localtime1 = time.asctime( time.localtime(time.time()) )
print ("本地时间为 :", localtime1)
nb_lstm_outputs = 80  #神经元个数
nb_time_steps = 200  #时间序列长度
nb_input_vector = 90 #输入序列
ww=1
def read_datamid_huodong():
    k=4
    fn1 = '/content/drive/MyDrive/huodong/data/data.csv'
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
        temp_domain_label = np.tile(i, (90,))
        if i==0:
            train_domain_label=temp_domain_label
        else:
            train_domain_label = np.concatenate((train_domain_label, temp_domain_label), axis=0)
    print(train_domain_label.shape)#(162000,)
    train_domain_label=np_utils.to_categorical(train_domain_label)
    #train_domain_label = train_domain_label.reshape(810,200,9)
    print(train_domain_label.shape)#(162000,9)
    print(train_domain_label.shape)

    fn2 = '/content/drive/MyDrive/huodong/data/label.csv'
    csvdata = pd.read_csv(fn2, header=None)
    csvdata = np.array(csvdata, dtype=np.float64)
    print(csvdata.shape)  # (1200,6)
    temp_label = csvdata
    # for i in range(60):
    #     print(temp_label[i*20*200:(i+1)*20*200])
    train_label_ot=temp_label[(k-1)*120:k*120]
    train_label = np.concatenate((temp_label[:(k-1)*120], temp_label[k*120:]), axis=0)
    for i in range(54):
        idx1 = np.array([j for j in range((i * 20), (i * 20 + 8), 1)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range((i * 20 + 13), (i * 20 + 20), 1)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        idxt = np.array([j for j in range((i * 20 + 8), (i * 20 + 13), 1)])
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

    for i in range(6):
        idx1 = np.array([j for j in range((i * 20) * 200, (i * 20 + 1) * 200, 1)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range((i * 20 + 10) * 200, (i * 20 + 11) * 200, 1)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        idx1 = np.array([j for j in range((i * 20 + 1) * 200, (i * 20 + 10) * 200, 1)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range((i * 20 + 11) * 200, (i * 20 + 20) * 200, 1)])  # 取中心点处左右分布数据
        idxt = np.hstack((idx1, idx2))
        # print(filename)
        if i == 0:
            temp_feature = train_feature_ot[idx]
            temp_feature2 = train_feature_ot[idxt]
        else:
            temp_feature = np.concatenate((temp_feature, train_feature_ot[idx]), axis=0)
            temp_feature2 = np.concatenate((temp_feature2, train_feature_ot[idxt]), axis=0)
    train_feature_ot_weitiao = temp_feature
    train_feature_ot = temp_feature2


    return np.array(train_feature), np.array(test_feature),np.array(train_feature_ot),np.array(train_domain_label),np.array(train_label),np.array(test_label),np.array(train_label_ot)
#对每个人的数据单独聚类
img_rows = 9
img_cols = 10
channels = 1
img_shape = (200,img_rows, img_cols, channels)

epochs = 5000
batch_size = 100
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
# a=np.concatenate((train_feature, test_feature), axis=0)
# train_feature = (train_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
# test_feature = (test_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
# train_feature_ot=(train_feature_ot.astype('float32')-np.min(a))/(np.max(a)-np.min(a))

#列归化为0~1
min_max_scaler = MinMaxScaler(feature_range=[0,1])
all = np.concatenate((train_feature, test_feature), axis=0)
all = np.concatenate((all, train_feature_ot), axis=0)
all= min_max_scaler.fit_transform(all)
train_feature = all[:len(train_feature)]
test_feature = all[len(train_feature):(len(train_feature)+len(test_feature))]
train_feature_ot = all[(len(train_feature)+len(test_feature)):]

#行归化为0~1
# min_max_scaler = MinMaxScaler(feature_range=[0,1])
# all = np.concatenate((train_feature, test_feature), axis=0)
# all = np.concatenate((all, train_feature_ot), axis=0)
# all=all.T
# all= min_max_scaler.fit_transform(all)
# all=all.T
# train_feature = all[:len(train_feature)]
# test_feature = all[len(train_feature):(len(train_feature)+len(test_feature))]
# train_feature_ot = all[(len(train_feature)+len(test_feature)):]

train_feature= train_feature.reshape([int(train_feature.shape[0]/200),200, img_rows, img_cols])
test_feature= test_feature.reshape([int(test_feature.shape[0]/200),200, img_rows, img_cols])
train_feature_ot= train_feature_ot.reshape([int(train_feature_ot.shape[0]/200),200, img_rows, img_cols])
train_feature = np.expand_dims(train_feature, axis=4)
test_feature = np.expand_dims(test_feature, axis=4)
train_feature_ot = np.expand_dims(train_feature_ot, axis=4)


latent_dim = 90
def build_cnn(img_shape):
    cnn = Sequential()
    cnn.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same',input_shape=img_shape)))
    cnn.add(BatchNormalization(epsilon=1e-6,axis=3))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3),activation='relu',strides=(1,1), padding='same')))
    cnn.add(BatchNormalization(epsilon=1e-6,axis=3))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same')))
    cnn.add(BatchNormalization(epsilon=1e-6,axis=3))
    cnn.add(TimeDistributed(Flatten()))
    cnn.add(TimeDistributed(Dense(180, activation="relu")))
    img = Input(shape=img_shape)
    latent_repr = cnn(img)
    return Model(img, latent_repr)
def build_rnn():
    rnn=Sequential()
    rnn.add(Bidirectional(LSTM(units=120, input_shape=(nb_time_steps, nb_input_vector))))
    rnn.add(Dense(500, activation="relu"))
    rnn.add(Dense(6, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = rnn(encoded_repr)
    return Model(encoded_repr, validity)
def build_dis():
    dis = Sequential()
    dis.add(Bidirectional(LSTM(units=120, input_shape=(nb_time_steps, nb_input_vector))))
    dis.add(Dense(500, activation="relu"))
    dis.add(Dense(9, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = dis(encoded_repr)
    return Model(encoded_repr, validity)

opt = adam_v2.Adam(0.0002, 0.5)
cnn = build_cnn(img_shape)
rnn=build_rnn()
dis = build_dis()
rnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
dis.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

img3 = Input(shape=img_shape)
encoded_repr3 = cnn(img3)
def get_class(x):
    return x[:,:,:latent_dim]
def get_dis(x):
    return x[:,:,latent_dim:]
encoded_repr3_class = Lambda(get_class)(encoded_repr3)
encoded_repr3_dis = Lambda(get_dis)(encoded_repr3)
validity1 = rnn(encoded_repr3_class)
validity2 = dis(encoded_repr3_dis)
crnn_model=Model(img3,validity1)
crnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
dis_model=Model(img3,validity2)
dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# crnn_model.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000crnn_model.h5')
# dis.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000dis.h5')
k=0
for epoch in range(epochs):

    idx = np.random.randint(0, train_feature.shape[0], batch_size)
    imgs = train_feature[idx]
    d_loss = dis_model.train_on_batch(imgs, train_domain_label[idx])
    crnn_loss = crnn_model.train_on_batch(imgs, train_label[idx])

    if epoch % 20 == 0:
        print("%d [活动分类loss: %f,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,]" % (epoch, crnn_loss[0], 100 * crnn_loss[1],d_loss[0], 100 * d_loss[1]))
        n = 0
        a_all=np.zeros((6, 6))
        for o in range(9):
            non_mid = crnn_model.predict(test_feature[o*5*6:(o+1)*5*6])
            non_pre = non_mid
            m = 0
            a = np.zeros((6, 6))
            for i in range(6):
                for k in range(5):
                    x = np.argmax(non_pre[i * 5 + k])
                    a[i][x]=a[i][x]+1
                    a_all[i][x] = a_all[i][x] + 1
                    if (x == i):
                        m = m + 1
                        n=n+1
            acc = float(m) / float(len(non_pre))
            print("源"+str(o+1)+"测试数据准确率：" + str(acc))
            #print(a)
        ac = float(n) / float(270)
        k1=ac
        print("源平均测试数据准确率：" + str(ac))
        print(a_all)


        non_mid = crnn_model.predict(train_feature_ot)
        non_pre =non_mid
        m = 0
        b=np.zeros((6, 6))
        for i in range(6):
            for k in range(20):
                x = np.argmax(non_pre[i * 20 + k])
                b[i][x] = b[i][x] + 1
                if (x == i):
                    m = m + 1
        acc = float(m) / float(len(non_pre))
        print("目标测试数据准确率：" + str(acc))
        print(b)
        kk1 = acc



        # if ((acc_non_pre3_vot >= 0.66) and (acc_yes_pre3_vot >= 0.66) and (c_loss[1] >= 0.65) and (
        #         acc_non_pre4_vot >= 0.66) and (acc_yes_pre4_vot >= 0.66)):
    #     if ((k1 >= 0.90) and(kk1 >= 0.35)and(d_loss[1] >= 0.90)):
    #         k1 = k1 * 1000
    #         k1 = int(k1)

    #         kk1 = kk1 * 1000
    #         kk1 = int(kk1)
    #         c = 100 * crnn_loss[1]
    #         c = int(c)
    #         d = 100 * d_loss[1]
    #         d = int(d)
    #         file = r'/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/result_crnn-dislstm-to-4.txt'
    #         f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
    #         str1 = str(epoch) + 'mid_' + str(c) +'d' + str(d) +'y_' + str(
    #             k1)+ 'm' + str(kk1)+ '\n'
    #         f.write(str1.encode())
    #         f.close()  # 关闭文件
    # if epoch == 1000:
    #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/1000crnn_model.h5')
    #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/1000dis.h5')
    # if epoch == 2000:
    #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/2000crnn_model.h5')
    #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/2000dis.h5')
    # if epoch == 3000:
    #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/3000crnn_model.h5')
    #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/3000dis.h5')
    # if epoch == 4000:
    #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000crnn_model.h5')
    #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000dis.h5')


localtime2 = time.asctime( time.localtime(time.time()) )
print ("开始时间为 :", localtime1)
print ("结束时间为 :", localtime2)