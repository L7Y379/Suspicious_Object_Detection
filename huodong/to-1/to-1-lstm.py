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
from keras.layers import Lambda,Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils
import time
localtime1 = time.asctime( time.localtime(time.time()) )
print ("本地时间为 :", localtime1)
nb_lstm_outputs = 80  #神经元个数
nb_time_steps = 200  #时间序列长度
nb_input_vector = 90 #输入序列
ww=1
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

    return np.array(train_feature), np.array(test_feature),np.array(train_feature_ot),np.array(train_domain_label),np.array(train_label),np.array(test_label),np.array(train_label_ot)
#对每个人的数据单独聚类
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
# def build_class(latent_dim):
#     model = Sequential()
#     model.add(Dense(400, input_dim=latent_dim, activation="relu"))
#     model.add(Dense(400, activation="relu"))
#     model.add(Dense(400, activation="relu"))
#     model.add(Dense(6, activation="softmax"))
#     encoded_repr = Input(shape=(latent_dim,))
#     validity = model(encoded_repr)
#     return Model(encoded_repr, validity)
def build_lstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(units=80, input_shape=(nb_time_steps, nb_input_vector))))
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
lstm = build_lstm()
lstm.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
validity2 = dis(encoded_repr3_dis)
dis_model=Model(img3,validity2)
dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

k=0
for epoch in range(epochs):

    # ---------------------
    #  Train lstm
    # ---------------------

    # Select a random batch of images
    idx2 = np.random.randint(0, train_feature.shape[0], batch_size)
    imgs2 = train_feature[idx2]
    sc_fido_loss = sc_fido.train_on_batch(imgs2, imgs2)
    idx = np.random.randint(0, train_feature.shape[0], batch_size)
    imgs = train_feature[idx]
    d_loss = dis_model.train_on_batch(imgs, train_domain_label[idx])

    latent_mid = ed.predict(train_feature)
    latent_mid = latent_mid[:, :latent_dim]
    latent_mid = latent_mid.reshape(810, nb_time_steps, nb_input_vector)
    idx = np.random.randint(0, latent_mid.shape[0], 400)
    c_loss = lstm.train_on_batch(latent_mid[idx], train_label[idx])

    if epoch % 10 == 0:
        print("%d [活动分类loss: %f,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,重构loss: %f]" % (
            epoch, c_loss[0], 100 * c_loss[1], d_loss[0], 100 * d_loss[1], sc_fido_loss))
        n = 0
        for o in range(9):
            non_mid = ed.predict(test_feature[o*5*6*200:(o+1)*5*6*200])
            non_mid = non_mid[:, :latent_dim]
            non_mid = non_mid.reshape(30, nb_time_steps, nb_input_vector)
            non_pre = lstm.predict(non_mid)
            m = 0
            for i in range(6):
                for k in range(5):
                    x = np.argmax(non_pre[i * 5 + k])
                    if (x == i):
                        m = m + 1
                        n=n+1
            acc = float(m) / float(len(non_pre))
            print("源"+str(o+1)+"测试数据准确率：" + str(acc))
        ac = float(n) / float(270)
        k1=ac
        print("源平均测试数据准确率：" + str(ac), end='   ')
        print()

        non_mid = ed.predict(train_feature_ot)
        non_mid = non_mid[:, :latent_dim]
        non_mid = non_mid.reshape(120, nb_time_steps, nb_input_vector)
        non_pre =lstm.predict(non_mid)
        m = 0
        for i in range(6):
            for k in range(20):
                x = np.argmax(non_pre[i * 20 + k])
                if (x == i):
                    m = m + 1
        acc = float(m) / float(len(non_pre))
        print("目标测试数据准确率：" + str(acc), end='   ')
        kk1 = acc
        print()


        # if ((acc_non_pre3_vot >= 0.66) and (acc_yes_pre3_vot >= 0.66) and (c_loss[1] >= 0.65) and (
        #         acc_non_pre4_vot >= 0.66) and (acc_yes_pre4_vot >= 0.66)):
        if ((k1 >= 0.65) and(kk1 >= 0.65)):
            k1 = k1 * 1000
            k1 = int(k1)

            kk1 = kk1 * 1000
            kk1 = int(kk1)
            c = 100 * c_loss[1]
            c = int(c)
            file = r'models/lstm/result_dingwei.txt'
            f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
            str1 = str(epoch) + 'mid_' + str(c) + 'y_' + str(
                k1)+ 'm' + str(kk1)+ '\n'
            f.write(str1.encode())
            f.close()  # 关闭文件
    if epoch == 500:
        lstm.save_weights('models/lstm/500lstm.h5')
        ed.save_weights('models/lstm/500ed.h5')
        dd.save_weights('models/lstm/500dd.h5')
        dis.save_weights('models/lstm/500dis.h5')
    if epoch == 1000:
        lstm.save_weights('models/lstm/1000lstm.h5')
        ed.save_weights('models/lstm/1000ed.h5')
        dd.save_weights('models/lstm/1000dd.h5')
        dis.save_weights('models/lstm/1000dis.h5')

    if epoch == 2000:
        lstm.save_weights('models/lstm/2000lstm.h5')
        ed.save_weights('models/lstm/2000ed.h5')
        dd.save_weights('models/lstm/2000dd.h5')
        dis.save_weights('models/lstm/2000dis.h5')

    if epoch == 3000:
        lstm.save_weights('models/lstm/3000lstm.h5')
        ed.save_weights('models/lstm/3000ed.h5')
        dd.save_weights('models/lstm/3000dd.h5')
        dis.save_weights('models/lstm/3000dis.h5')

    if epoch == 4000:
        lstm.save_weights('models/lstm/4000lstm.h5')
        ed.save_weights('models/lstm/4000ed.h5')
        dd.save_weights('models/lstm/4000dd.h5')
        dis.save_weights('models/lstm/4000dis.h5')

lstm.save_weights('models/lstm/lstm.h5')
ed.save_weights('models/lstm/ed.h5')
dd.save_weights('models/lstm/dd.h5')
dis.save_weights('models/lstm/dis.h5')

localtime2 = time.asctime( time.localtime(time.time()) )
print ("开始时间为 :", localtime1)
print ("结束时间为 :", localtime2)