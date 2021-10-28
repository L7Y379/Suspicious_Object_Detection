import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
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


def read_data():
    filepath = '/content/drive/MyDrive/data/use_data/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['ori_gzy']:
        fn = filepath + name + "/train" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature = csvdata[:, 0:270]
        train_feature_reg = csvdata[:, 270:]
        print(train_feature.shape)
        for i in range(24):
            idx1 = np.array([j for j in range(i * 100, i * 100 + 40, 1)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(i * 100 + 60, i * 100 + 100, 1)])  # 取中心点处左右分布数据
            idx = np.hstack((idx1, idx2))
            idxt = np.array([j for j in range(i * 100 + 40, i * 100 + 60, 1)])
            # print(filename)
            if i == 0:
                temp_feature = train_feature[idx]
                temp_feature2 = train_feature[idxt]
            else:
                temp_feature = np.concatenate((temp_feature, train_feature[idx]), axis=0)
                temp_feature2 = np.concatenate((temp_feature2, train_feature[idxt]), axis=0)

        train_feature = temp_feature
        test_feature = temp_feature2
        print(train_feature.shape)
        print(test_feature.shape)

        train_label_reg = np.arange(24 * 2)
        train_label_reg = train_label_reg.reshape(24, 2)
        for i in range(24):
            train_label_reg[i] = train_feature_reg[i * 100 + 1]
        print('train_label_reg' + str(train_label_reg))
        for i in range(24):
            label = np.tile(i, (80,))
            label2 = np.tile(i, (20,))
            if (i == 0):
                train_label = label
                test_label = label2
            else:
                train_label = np.concatenate((train_label, label), axis=0)
                test_label = np.concatenate((test_label, label2), axis=0)
        print(train_label.shape)
        print(train_label)
        print(test_label.shape)
        print(test_label)
        temp_label = np.tile(0, (train_label.shape[0],))
        print("temp_label" + str(temp_label.shape))
        print(temp_label)

    for name in ['ori_tk']:
        fn = filepath + name + "/train" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature2 = csvdata[:, 0:270]
        print(train_feature2.shape)

        for i in range(24):
            idx1 = np.array([j for j in range(i * 100, i * 100 + 40, 1)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(i * 100 + 60, i * 100 + 100, 1)])  # 取中心点处左右分布数据
            idx = np.hstack((idx1, idx2))
            idxt = np.array([j for j in range(i * 100 + 40, i * 100 + 60, 1)])
            # print(filename)
            if i == 0:
                temp_feature = train_feature2[idx]
                temp_feature2 = train_feature2[idxt]
            else:
                temp_feature = np.concatenate((temp_feature, train_feature2[idx]), axis=0)
                temp_feature2 = np.concatenate((temp_feature2, train_feature2[idxt]), axis=0)

        train_feature = np.concatenate((train_feature, temp_feature), axis=0)
        test_feature = np.concatenate((test_feature, temp_feature2), axis=0)
        print(train_feature.shape)
        print(test_feature.shape)

        train_label = np.concatenate((train_label, train_label), axis=0)
        print(train_label[:1920].shape)
        print(train_label[:1920])
        print(train_label[1920:].shape)
        print(train_label[1920:])
        train_label = np_utils.to_categorical(train_label)
        print(train_label.shape)
        print(train_label)
        test_label = np.concatenate((test_label, test_label), axis=0)
        print(test_label.shape)
        print(test_label)
        test_label = np_utils.to_categorical(test_label)
        print(test_label.shape)
        print(test_label)
        temp_label2 = np.tile(1, (1920,))
        temp_label = np.concatenate((temp_label, temp_label2), axis=0)
        domain_label = np_utils.to_categorical(temp_label)
        print("temp_label" + str(temp_label.shape))
        print(temp_label)
        print("domain_label" + str(domain_label.shape))
        print(domain_label)

    return np.array(train_feature), np.array(train_label), np.array(domain_label), np.array(test_feature), np.array(
        test_label), np.array(train_label_reg)


def read_data420():
    filepath = 'D:/my bad/CSI_DATA/code/DaNN/420USEFUL/B/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['Train_Amti']:
        fn = filepath + name + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=0)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature = csvdata[:, 0:270]
        print(train_feature.shape)
        train_label = csvdata[:, 270:]
        print(train_label.shape)
        temp_label = np.tile(0, (1200,))

    for name in ['Test_Amti']:
        fn = filepath + name + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=0)
        csvdata = np.array(csvdata, dtype=np.float64)
        test_feature = csvdata[:, 0:270]
        print(test_feature.shape)
        test_label = csvdata[:, 270:]
        print(test_label.shape)
        temp_label2 = np.tile(1, (1200,))
        temp_label = np.concatenate((temp_label, temp_label2), axis=0)
        domain_label = np_utils.to_categorical(temp_label)

    return np.array(train_feature), np.array(train_label), np.array(domain_label), np.array(test_feature), np.array(
        test_label)


def read_data_ot():
    filepath = '/content/drive/MyDrive/data/use_data/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['ori_zb']:
        fn = filepath + name + "/train" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature = csvdata[:, 0:270]
        print(train_feature.shape)
        for i in range(24):
            label = np.tile(i, (100,))
            if (i == 0):
                train_label = label
            else:
                train_label = np.concatenate((train_label, label), axis=0)
        print(train_label.shape)
        print(train_label)
        train_label = np_utils.to_categorical(train_label)
        print(train_label.shape)
        print(train_label)

    return np.array(train_feature), np.array(train_label)


train_feature, train_label, train_domain_label, test_feature, test_label, train_label_reg = read_data()
train_feature_ot, train_label_ot = read_data_ot()
print("train_feature" + str(train_feature.shape))
print("train_label" + str(train_label.shape))
print("test_feature" + str(test_feature.shape))
print("test_label" + str(test_label.shape))
print("domain_label" + str(train_domain_label.shape))
print("train_feature_ot" + str(train_feature_ot.shape))
print("train_label_ot" + str(train_label_ot.shape))
img_rows = 15
img_cols = 18
channels = 1
img_shape = (img_rows, img_cols, channels)

epochs = 5000
batch_size = 3000

# a = train_feature
# train_feature = (train_feature.astype('float32') - np.min(a)) / (np.max(a) - np.min(a))
# test_feature = (test_feature.astype('float32') - np.min(a)) / (np.max(a) - np.min(a))
# train_feature_ot = (train_feature_ot.astype('float32') - np.min(a)) / (np.max(a) - np.min(a))

min_max_scaler = MinMaxScaler(feature_range=[0,1])
all = np.concatenate((train_feature, test_feature), axis=0)
all = np.concatenate((all, train_feature_ot), axis=0)
all= min_max_scaler.fit_transform(all)

train_feature = all[:len(train_feature)]
test_feature = all[len(train_feature):(len(train_feature)+len(test_feature))]
train_feature_ot = all[(len(train_feature)+len(test_feature)):]

train_feature = train_feature.reshape([train_feature.shape[0], img_rows, img_cols])
train_feature = np.expand_dims(train_feature, axis=3)
test_feature = test_feature.reshape([test_feature.shape[0], img_rows, img_cols])
test_feature = np.expand_dims(test_feature, axis=3)
train_feature_ot = train_feature_ot.reshape([train_feature_ot.shape[0], img_rows, img_cols])
train_feature_ot = np.expand_dims(train_feature_ot, axis=3)

latent_dim = 270
latent_dim2 = 540


def build_ed(latent_dim2, img_shape):
    deterministic = 1
    img = Input(shape=img_shape)
    h = Flatten()(img)
    h = Dense(800, activation="relu")(h)
    h = Dense(800, activation="relu")(h)
    h = Dense(800, activation="relu")(h)
    latent_repr = Dense(latent_dim2)(h)
    return Model(img, latent_repr)


def build_class(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(24, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)


def build_class2(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(24, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)


def build_dis(latent_dim):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)


def build_dd(latent_dim2, img_shape):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim2, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(np.prod(img_shape), activation='sigmoid'))
    model.add(Reshape(img_shape))
    z = Input(shape=(latent_dim2,))
    img = model(z)
    return Model(z, img)


opt = Adam(0.0002, 0.5)
classer = build_class(latent_dim)
classer.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
classer2 = build_class2(latent_dim)
classer2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
dis = build_dis(latent_dim)
dis.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
ed = build_ed(latent_dim2, img_shape)
dd = build_dd(latent_dim2, img_shape)

# img3 = Input(shape=img_shape)
# encoded_repr3 = ed(img3)
# reconstructed_img3 = dd(encoded_repr3)
# sc_fido = Model(img3,reconstructed_img3)
# sc_fido.compile(loss='mse', optimizer=opt)
# def get_class(x):
#     return x[:,:latent_dim]
# def get_dis(x):
#     return x[:,latent_dim:]
# encoded_repr3_class = Lambda(get_class)(encoded_repr3)
# encoded_repr3_dis = Lambda(get_dis)(encoded_repr3)
# validity1 = classer(encoded_repr3_class)
# validity2 = dis(encoded_repr3_dis)
# dis_model=Model(img3,validity2)
# dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


img3 = Input(shape=img_shape)
encoded_repr3 = ed(img3)
reconstructed_img3 = dd(encoded_repr3)
sc_fido = Model(img3, reconstructed_img3)
sc_fido.compile(loss='mse', optimizer=opt)


def get_class(x):
    return x[:, :latent_dim]


def get_dis(x):
    return x[:, latent_dim:]


encoded_repr3_class = Lambda(get_class)(encoded_repr3)
encoded_repr3_dis = Lambda(get_dis)(encoded_repr3)
validity1 = classer(encoded_repr3_class)
validity1_2 = classer2(encoded_repr3_class)
validity2 = dis(encoded_repr3_dis)
class_model = Model(img3, validity1)
class_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
class_model2 = Model(img3, validity1_2)
class_model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
dis_model = Model(img3, validity2)
dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

all_data = np.concatenate((train_feature, train_feature_ot), axis=0)
train_label1 = train_label[:int(len(train_label) / 2)]
train_label2 = train_label[int(len(train_label) / 2):]
print(train_label1.shape)
print(train_label2.shape)
train_feature1 = train_feature[:int(len(train_feature) / 2)]
train_feature2 = train_feature[int(len(train_feature) / 2):]
print(train_feature1.shape)
print(train_feature2.shape)
# kmeans1 = KMeans(n_clusters=1, n_init=50)
# kmeans2 = KMeans(n_clusters=1, n_init=50)
# pred_train1 = kmeans1.fit_predict(train_feature1)
# pred_train2 = kmeans2.fit_predict(train_feature2)
# kmeans1.cluster_centers_
# classer.load_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000classer.h5')
# classer2.load_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000classer2.h5')
# ed.load_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000ed.h5')
# dd.load_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000dd.h5')
# dis.load_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000dis.h5')
k = 0
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

    idx3 = np.random.randint(0, train_feature1.shape[0], int(batch_size / 2))
    imgs1 = train_feature1[idx3]
    imgs2 = train_feature2[idx3]

    c_loss1 = class_model.train_on_batch(imgs1, train_label1[idx3])
    c_loss2 = class_model2.train_on_batch(imgs2, train_label2[idx3])

    # latent_mid1 = ed.predict(imgs1)
    # latent_mid1 = latent_mid1[:, :latent_dim]
    # c_loss1 = classer.train_on_batch(latent_mid1, train_label1[idx3])
    #
    # latent_mid2 = ed.predict(imgs2)
    # latent_mid2 = latent_mid2[:, :latent_dim]
    # c_loss2 = classer2.train_on_batch(latent_mid2, train_label2[idx3])
    # ---------------------
    #  Train dis
    # ---------------------

    # ---------------------
    #  Train chonggou
    # ---------------------

    # Plot the progress (every 10th epoch)
    if epoch % 5 == 0:
        print("%d [危险品分类acc: %.2f%%,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,重构loss: %f]" % (
            epoch, 100 * c_loss1[1], 100 * c_loss2[1], d_loss[0], 100 * d_loss[1], sc_fido_loss))

        non_mid = ed.predict(test_feature[:480])
        non_mid = non_mid[:, :latent_dim]
        non_pre = classer.predict(non_mid)
        yes_mid = ed.predict(test_feature[480:])
        yes_mid = yes_mid[:, :latent_dim]
        yes_pre = classer2.predict(yes_mid)
        m = 0
        n = 0
        for i in range(24):
            for k in range(20):
                x = np.argmax(non_pre[i * 20 + k])
                if (x == i):
                    m = m + 1
                    n = n + 0
                else:
                    a = train_label_reg[x] - train_label_reg[i]
                    a = np.power(a, 2)
                    a = np.sum(a)
                    a = np.sqrt(a)
                    n = n + a
        ac = float(n) / float(480)
        acc = float(m) / float(len(non_pre))
        k1=acc
        kk1=ac
        print("源1测试数据准确率：" + str(acc) + "    精度：" + str(ac))

        m = 0
        n = 0
        for i in range(24):
            for k in range(20):
                x = np.argmax(yes_pre[i * 20 + k])
                if (x == i):
                    m = m + 1
                    n = n + 0
                else:
                    a = train_label_reg[x] - train_label_reg[i]
                    a = np.power(a, 2)
                    a = np.sum(a)
                    a = np.sqrt(a)
                    n = n + a
        ac = float(n) / float(480)
        acc = float(m) / float(len(yes_pre))
        k2 = acc
        kk2=ac
        print("源2测试数据准确率：" + str(acc) + "    精度：" + str(ac))

        non_mid = ed.predict(train_feature_ot[:2400])
        non_mid = non_mid[:, :latent_dim]
        non_pre = classer.predict(non_mid)
        yes_mid = ed.predict(train_feature_ot[:2400])
        yes_mid = yes_mid[:, :latent_dim]
        yes_pre = classer2.predict(yes_mid)

        m = 0
        for i in range(24):
            for k in range(100):
                if (np.argmax(non_pre[i * 100 + k]) == i):
                    m = m + 1
        acc = float(m) / float(len(non_pre))
        print("他人1训练数准确率：" + str(acc))
        m = 0
        for i in range(24):
            for k in range(100):
                if (np.argmax(yes_pre[i * 100 + k]) == i):
                    m = m + 1
        acc = float(m) / float(len(yes_pre))
        print("他人1训练数准确率：" + str(acc))

        dis_mid = ed.predict(train_feature_ot[:2400])
        dis_mid = dis_mid[:, latent_dim:]
        dis_pre = dis.predict(dis_mid)
        dis_pre1 = dis_pre[:, :1]
        dis_pre2 = dis_pre[:, 1:]
        print(dis_pre)
        non_pre = non_pre
        yes_pre = yes_pre
        non_pre = non_pre * dis_pre1
        yes_pre = yes_pre * dis_pre2
        pre = non_pre + yes_pre
        m = 0
        n = 0
        for i in range(24):
            for k in range(100):
                x = np.argmax(pre[i * 100 + k])
                if (x == i):
                    m = m + 1
                    n = n + 0
                else:
                    a = train_label_reg[x] - train_label_reg[i]
                    a = np.power(a, 2)
                    a = np.sum(a)
                    a = np.sqrt(a)
                    n = n + a
        ac = float(n) / float(2400)
        acc = float(m) / float(len(pre))
        k3=acc
        kk3=ac
        print("他人1训练数准确率：" + str(acc) + "    精度：" + str(ac))
        print()
        if ((k1 >= 0.93) and (k2 >= 0.93)and (kk3 <= 140)):
            k1 = k1 * 1000
            k1 = int(k1)
            k2 = k2 * 1000
            k2 = int(k2)
            k3 = k3 * 1000
            k3 = int(k3)

            kk1 = int(kk1)
            kk2 = int(kk2)
            kk3 = int(kk3)
            d = 100 * d_loss[1]
            d = int(d)
            file =r'/content/drive/MyDrive/AAAA/result_dingwei.txt'
            f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
            str1 = str(epoch) + 'mid_' + str(d) + 'y1_' + str(
                k1) + '_' + str(kk1) + '_y2_' + str(k2) + '_' + str(
                kk2) + 'm' + str(k3) + '_' + str(
                kk3)+ '\n'
            f.write(str1.encode())
            f.close()  # 关闭文件

    if epoch == 2000:
        classer.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000classer.h5')
        classer2.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000classer2.h5')
        ed.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000ed.h5')
        dd.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000dd.h5')
        dis.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/2000dis.h5')

print("%d [危险品分类acc: %.2f%%,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,重构loss: %f]" % (
    epoch, 100 * c_loss1[1], c_loss2[1], d_loss[0], 100 * d_loss[1], sc_fido_loss))
classer.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/classer.h5')
classer2.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/classer2.h5')
ed.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/ed.h5')
dd.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/dd.h5')
dis.save_weights('/content/drive/MyDrive/AAAA/models/dingwei/dis.h5')