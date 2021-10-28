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
    filepath = 'D:/my bad/Suspicious object detection/data/use_data/'
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
        train_feature_reg=csvdata[:, 270:]
        print(train_feature.shape)
        for i in range(24):
            idx1 = np.array([j for j in range(i*100, i*100+40, 1)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(i*100+60, i*100+100, 1)])  # 取中心点处左右分布数据
            idx = np.hstack((idx1, idx2))
            idxt = np.array([j for j in range(i*100+40, i*100+60, 1)])
            # print(filename)
            if i==0:
                temp_feature = train_feature[idx]
                temp_feature2 = train_feature[idxt]
            else:
                temp_feature = np.concatenate((temp_feature, train_feature[idx]), axis=0)
                temp_feature2 = np.concatenate((temp_feature2, train_feature[idxt]), axis=0)

        train_feature = temp_feature
        test_feature = temp_feature2
        print(train_feature.shape)
        print(test_feature.shape)

        train_label_reg=np.arange(24*2)
        train_label_reg=train_label_reg.reshape(24,2)
        for i in range(24):
            train_label_reg[i] = train_feature_reg[i*100+1]
        print('train_label_reg'+str(train_label_reg))

        fn = filepath + name + "/test" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        test_feature_12 = csvdata[:, 0:270]
        train_feature_reg2 = csvdata[:, 270:]
        print(test_feature_12.shape)
        train_label_reg2 = np.arange(12 * 2)
        train_label_reg2 = train_label_reg2.reshape(12, 2)
        for i in range(12):
            train_label_reg2[i] = train_feature_reg2[i * 100 + 1]
        print('train_label_reg2' + str(train_label_reg2))

        for i in range(24):
            label = np.tile(i, (80,))
            label2 = np.tile(i, (20,))
            if (i ==0):
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
            idx1 = np.array([j for j in range(i*100, i*100+40, 1)])  # 取中心点处左右分布数据
            idx2 = np.array([j for j in range(i*100+60, i*100+100, 1)])  # 取中心点处左右分布数据
            idx = np.hstack((idx1, idx2))
            idxt = np.array([j for j in range(i*100+40, i*100+60, 1)])
            # print(filename)
            if i==0:
                temp_feature = train_feature2[idx]
                temp_feature2 = train_feature2[idxt]
            else:
                temp_feature = np.concatenate((temp_feature, train_feature2[idx]), axis=0)
                temp_feature2 = np.concatenate((temp_feature2, train_feature2[idxt]), axis=0)

        train_feature = np.concatenate((train_feature, temp_feature), axis=0)
        test_feature = np.concatenate((test_feature, temp_feature2), axis=0)
        print(train_feature.shape)
        print(test_feature.shape)

        fn = filepath + name + "/test" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        test_feature_12 = np.concatenate((test_feature_12, csvdata[:, 0:270]), axis=0)
        print(test_feature_12.shape)

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
        temp_label= np.concatenate((temp_label, temp_label2), axis=0)
        domain_label = np_utils.to_categorical(temp_label)
        print("temp_label" + str(temp_label.shape))
        print(temp_label)
        print("domain_label"+str(domain_label.shape))
        print(domain_label)

    return np.array(train_feature), np.array(train_label),np.array(domain_label),np.array(test_feature),np.array(test_label),np.array(train_label_reg),np.array(test_feature_12),np.array(train_label_reg2)
def read_data_ot():
    filepath = 'D:/my bad/Suspicious object detection/data/use_data/'
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
            if (i ==0):
                train_label = label
            else:
                train_label = np.concatenate((train_label, label), axis=0)
        print(train_label.shape)
        print(train_label)
        train_label = np_utils.to_categorical(train_label)
        print(train_label.shape)
        print(train_label)

        fn = filepath + name + "/test" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature_12 = csvdata[:, 0:270]

    return np.array(train_feature), np.array(train_label),np.array(train_feature_12)
train_feature, train_label,train_domain_label,test_feature,test_label, train_label_reg,test_feature_12,train_label_reg2= read_data()
train_feature_ot, train_label_ot,train_feature_ot_12 = read_data_ot()
print("train_feature"+str(train_feature.shape))
print("train_label"+str(train_label.shape))
print("test_feature"+str(test_feature.shape))
print("test_label"+str(test_label.shape))
print("domain_label"+str(train_domain_label.shape))
print("train_feature_ot"+str(train_feature_ot.shape))
print("train_label_ot"+str(train_label_ot.shape))
print("test_feature_12"+str(test_feature_12.shape))
print("train_label_reg2"+str(train_label_reg2.shape))
print("train_feature_ot_12"+str(train_feature_ot_12.shape))
img_rows = 15
img_cols = 18
channels = 1
img_shape = (img_rows, img_cols, channels)

epochs = 8000
batch_size = 3000

a=train_feature
train_feature = (train_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
test_feature = (test_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
train_feature_ot=(train_feature_ot.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
test_feature_12=(test_feature_12.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
train_feature_ot_12=(train_feature_ot_12.astype('float32')-np.min(a))/(np.max(a)-np.min(a))

# train_feature = (train_feature.astype('float32')-np.min(train_feature))/(np.max(train_feature)-np.min(train_feature))
# test_feature = (test_feature.astype('float32')-np.min(test_feature))/(np.max(test_feature)-np.min(test_feature))
# train_feature_ot=(train_feature_ot.astype('float32')-np.min(train_feature_ot))/(np.max(train_feature_ot)-np.min(train_feature_ot))

# min_max_scaler = MinMaxScaler(feature_range=[0,1])
# all = np.concatenate((train_feature, test_feature_12), axis=0)
# all = np.concatenate((all, train_feature_ot_12), axis=0)
# all= min_max_scaler.fit_transform(all)
#
# train_feature = all[:len(train_feature)]
# test_feature = all[len(train_feature):(len(train_feature)+len(test_feature))]
# train_feature_ot = all[(len(train_feature)+len(test_feature)):]

train_feature = train_feature.reshape([train_feature.shape[0], img_rows, img_cols])
train_feature = np.expand_dims(train_feature, axis=3)
test_feature = test_feature.reshape([test_feature.shape[0], img_rows, img_cols])
test_feature = np.expand_dims(test_feature, axis=3)
train_feature_ot = train_feature_ot.reshape([train_feature_ot.shape[0], img_rows, img_cols])
train_feature_ot = np.expand_dims(train_feature_ot, axis=3)
test_feature_12 = test_feature_12.reshape([test_feature_12.shape[0], img_rows, img_cols])
test_feature_12 = np.expand_dims(test_feature_12, axis=3)
train_feature_ot_12 = train_feature_ot_12.reshape([train_feature_ot_12.shape[0], img_rows, img_cols])
train_feature_ot_12 = np.expand_dims(train_feature_ot_12, axis=3)


latent_dim = 270
latent_dim2=540

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


classer.load_weights('models/class-ot12/1000classer.h5')
ed.load_weights('models/class-ot12/1000ed.h5')
dd.load_weights('models/class-ot12/1000dd.h5')
dis.load_weights('models/class-ot12/1000dis.h5')
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
    # ---------------------
    #  Train dis
    # ---------------------


    # ---------------------
    #  Train chonggou
    # ---------------------


    # Plot the progress (every 10th epoch)
    if epoch % 5 == 0:
        print("%d [危险品分类loss: %f,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,重构loss: %f]" % (
            epoch, c_loss[0], 100 * c_loss[1], d_loss[0], 100 * d_loss[1], sc_fido_loss))

        # non_mid = ed.predict(test_feature[:480])
        # non_mid = non_mid[:, :latent_dim]
        # non_pre = classer.predict(non_mid)
        # yes_mid = ed.predict(test_feature[480:])
        # yes_mid = yes_mid[:, :latent_dim]
        # yes_pre = classer.predict(yes_mid)
        # m = 0
        # n = 0
        # for i in range(24):
        #     for k in range(20):
        #         x = np.argmax(non_pre[i * 20 + k])
        #         if (x == i):
        #             m = m + 1
        #             n = n + 0
        #         else:
        #             a = train_label_reg[x] - train_label_reg[i]
        #             a = np.power(a, 2)
        #             a = np.sum(a)
        #             a = np.sqrt(a)
        #             n = n + a
        # ac = float(n) / float(480)
        # acc = float(m) / float(len(non_pre))
        # k1 = acc
        # kk1 = ac
        # print("源1测试数据准确率：" + str(acc)+"    精度："+str(ac))
        #
        # m = 0
        # n = 0
        # for i in range(24):
        #     for k in range(20):
        #         x = np.argmax(yes_pre[i * 20 + k])
        #         if (x == i):
        #             m = m + 1
        #             n = n + 0
        #         else:
        #             a = train_label_reg[x] - train_label_reg[i]
        #             a = np.power(a, 2)
        #             a = np.sum(a)
        #             a = np.sqrt(a)
        #             n = n + a
        # ac = float(n) / float(480)
        # acc = float(m) / float(len(yes_pre))
        # k2 = acc
        # kk2 = ac
        # print("源2测试数据准确率：" + str(acc)+"    精度："+str(ac))

        non_mid = ed.predict(test_feature_12[:1200])
        non_mid = non_mid[:, :latent_dim]
        non_pre = classer.predict(non_mid)
        yes_mid = ed.predict(test_feature_12[1200:])
        yes_mid = yes_mid[:, :latent_dim]
        yes_pre = classer.predict(yes_mid)
        n = 0
        for i in range(12):
            for k in range(100):
                x = np.argmax(non_pre[i * 100 + k])
                a = train_label_reg[x] - train_label_reg2[i]
                a = np.power(a, 2)
                a = np.sum(a)
                a = np.sqrt(a)
                n = n + a
        ac = float(n) / float(1200)
        kk1 = ac
        print("源1测试数据精度：" + str(ac))

        n = 0
        for i in range(12):
            for k in range(100):
                x = np.argmax(yes_pre[i * 100 + k])
                a = train_label_reg[x] - train_label_reg2[i]
                a = np.power(a, 2)
                a = np.sum(a)
                a = np.sqrt(a)
                n = n + a
        ac = float(n) / float(1200)
        kk2 = ac
        print("源2测试数据精度：" + str(ac))


        # non_mid = ed.predict(train_feature_ot[:2400])
        # non_mid = non_mid[:, :latent_dim]
        # non_pre = classer.predict(non_mid)
        #
        # m = 0
        # n = 0
        # for i in range(24):
        #     for k in range(100):
        #         x = np.argmax(non_pre[i * 100 + k])
        #         if (x == i):
        #             m = m + 1
        #             n = n + 0
        #         else:
        #             a = train_label_reg[x] - train_label_reg[i]
        #             a = np.power(a, 2)
        #             a = np.sum(a)
        #             a = np.sqrt(a)
        #             n = n + a
        # ac = float(n) / float(2400)
        # acc = float(m) / float(len(non_pre))
        # k3 = acc
        # kk3 = ac
        # print("他人1训练数准确率：" + str(acc)+"    精度："+str(ac))

        non_mid = ed.predict(train_feature_ot_12[:1200])
        non_mid = non_mid[:, :latent_dim]
        non_pre = classer.predict(non_mid)

        n = 0
        for i in range(12):
            for k in range(100):
                x = np.argmax(non_pre[i * 100 + k])
                a = train_label_reg[x] - train_label_reg2[i]
                a = np.power(a, 2)
                a = np.sum(a)
                a = np.sqrt(a)
                n = n + a
        ac = float(n) / float(1200)
        kk3 = ac
        print("他人1训练数精度：" + str(ac))
        print()
        if ((kk1 <= 155) and (kk2 <= 155) and (kk3 <= 250)):
            kk1 = int(kk1)
            kk2 = int(kk2)
            kk3 = int(kk3)
            d = 100 * d_loss[1]
            d = int(d)
            file = r'models/class-ot12/result_dingwei.txt'
            f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
            str1 = str(epoch) + 'mid_' + str(d) + 'y1_' + str(kk1) + '_y2_' + str(
                kk2) + 'm' + str(
                kk3) + '\n'
            f.write(str1.encode())
            f.close()  # 关闭文件

    if epoch == 1000:
        classer.save_weights('models/class-ot12/1000classer.h5')
        ed.save_weights('models/class-ot12/1000ed.h5')
        dd.save_weights('models/class-ot12/1000dd.h5')
        dis.save_weights('models/class-ot12/1000dis.h5')
    if epoch == 2000:
        classer.save_weights('models/class-ot12/2000classer.h5')
        ed.save_weights('models/class-ot12/2000ed.h5')
        dd.save_weights('models/class-ot12/2000dd.h5')
        dis.save_weights('models/class-ot12/2000dis.h5')
print("%d [危险品分类loss: %f,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,重构loss: %f]" % (
epoch, c_loss[0], 100 * c_loss[1],d_loss[0],100 * d_loss[1], sc_fido_loss))
classer.save_weights('models/class-ot12/classer.h5')
ed.save_weights('models/class-ot12/ed.h5')
dd.save_weights('models/class-ot12/dd.h5')
dis.save_weights('models/class-ot12/dis.h5')