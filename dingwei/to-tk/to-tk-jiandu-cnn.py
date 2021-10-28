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

        train_label_reg = np.arange(24 * 2)
        train_label_reg = train_label_reg.reshape(24, 2)
        for i in range(24):
            train_label_reg[i] = train_feature_reg[i * 100 + 1]
        print('train_label_reg' + str(train_label_reg))
        train_label = train_label_reg.repeat(80, axis=0)
        test_label = train_label_reg.repeat(20, axis=0)

        fn = filepath + name + "/test" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        test_feature_12 = csvdata[:, 0:270]
        print(test_feature_12.shape)
        test_label_12 = csvdata[:, 270:]
        print(test_label_12.shape)


    for name in ['ori_zb']:
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
        test_label_12 = np.concatenate((test_label_12, csvdata[:, 270:]), axis=0)
        train_label = np.concatenate((train_label, train_label), axis=0)
        test_label = np.concatenate((test_label, test_label), axis=0)

    return np.array(train_feature), np.array(train_label),np.array(test_feature),np.array(test_label),np.array(test_feature_12),np.array(test_label_12)
def read_data_ot():
    filepath = 'D:/my bad/Suspicious object detection/data/use_data/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['ori_tk']:
        fn = filepath + name + "/train" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature = csvdata[:, 0:270]
        train_label = csvdata[:, 270:]
        print(train_feature.shape)


        fn = filepath + name + "/test" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature_ot_12 = csvdata[:, 0:270]
        train_label_ot_12 = csvdata[:, 270:]
    return np.array(train_feature), np.array(train_label),np.array(train_feature_ot_12),np.array(train_label_ot_12)
train_feature, train_label,test_feature,test_label, test_feature_12,test_label_12= read_data()
train_feature_ot, train_label_ot,train_feature_ot_12,train_label_ot_12 = read_data_ot()
print("train_feature"+str(train_feature.shape))
print("train_label"+str(train_label.shape))
print("test_feature"+str(test_feature.shape))
print("test_label"+str(test_label.shape))
print("test_feature_12"+str(test_feature_12.shape))
print("test_label_12"+str(test_label_12.shape))
print("train_feature_ot"+str(train_feature_ot.shape))
print("train_label_ot"+str(train_label_ot.shape))
print("train_feature_ot_12"+str(train_feature_ot_12.shape))
print("train_label_ot_12"+str(train_label_ot_12.shape))
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
def build_class(latent_dim2):
    model = Sequential()
    model.add(Dense(800, input_dim=latent_dim2, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(2, activation="relu"))
    encoded_repr = Input(shape=(latent_dim2,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_cnn(latent_dim2, img_shape):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same',input_shape=img_shape))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',strides=(1,1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(latent_dim2, activation='relu'))
    model.add(Dense(latent_dim2, activation='relu'))
    model.add(Dense(2, activation='relu'))
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

opt = Adam(0.0002, 0.5)
cnn = build_cnn(latent_dim2, img_shape)
cnn.compile(loss='mse', optimizer=opt)
img3 = Input(shape=img_shape)
encoded_repr3 = cnn(img3)
cnn_model=Model(img3,encoded_repr3)
cnn_model.compile(loss='mse', optimizer=opt)

# classer.load_weights('models/jiandu/1000classer.h5')
# ed.load_weights('models/jiandu/1000ed.h5')
k=0
for epoch in range(epochs):

    # ---------------------
    #  Train classer
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, train_feature.shape[0], batch_size)
    imgs = train_feature[idx]
    c_loss = cnn_model.train_on_batch(imgs, train_label[idx])
    # ---------------------
    #  Train dis
    # ---------------------


    # ---------------------
    #  Train chonggou
    # ---------------------


    # Plot the progress (every 10th epoch)
    if epoch % 10 == 0:
        print("%d [定位损失loss: %f]" % (epoch, c_loss))

        non_mid = cnn.predict(train_feature[:1920])
        non_pre = non_mid
        yes_mid = cnn.predict(train_feature[1920:])
        yes_pre = yes_mid
        a = non_pre - train_label[:1920]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('源域1训练数据精度：' + str(a))
        a = yes_pre - train_label[1920:]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('源域2训练数据精度：' + str(a))

        non_mid = cnn.predict(test_feature[:480])
        non_pre = non_mid
        yes_mid = cnn.predict(test_feature[480:])
        yes_pre = yes_mid
        a = non_pre - test_label[:480]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('源域1测试数据精度：' + str(a))
        a = yes_pre - test_label[480:]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('源域2测试数据精度：' + str(a))


        non_mid = cnn.predict(test_feature_12[:1200])
        non_pre = non_mid
        yes_mid = cnn.predict(test_feature_12[1200:])
        yes_pre = yes_mid
        a = non_pre - test_label_12[:1200]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('12源域1测试数据精度：' + str(a))
        a = yes_pre - test_label_12[1200:]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('12源域2测试数据精度：' + str(a))


        non_mid = cnn.predict(train_feature_ot[:2400])
        non_pre = non_mid
        a = non_pre - train_label_ot[:2400]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('目域测试数据精度：' + str(a))

        non_mid = cnn.predict(train_feature_ot_12[:1200])
        non_pre = non_mid
        a = non_pre - train_label_ot_12[:1200]
        a = np.power(a, 2)
        a = np.sum(a, axis=1)
        a = np.sqrt(a)
        a = np.mean(a)
        print('12目域测试数据精度：' + str(a))
        print()
        # if ((kk1 <= 155) and (kk2 <= 155) and (kk3 <= 250)):
        #     kk1 = int(kk1)
        #     kk2 = int(kk2)
        #     kk3 = int(kk3)
        #     file = r'models/jiandu/result_dingwei.txt'
        #     f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
        #     str1 = str(epoch) + 'mid_' + str(d) + 'y1_' + str(kk1) + '_y2_' + str(
        #         kk2) + 'm' + str(
        #         kk3) + '\n'
        #     f.write(str1.encode())
        #     f.close()  # 关闭文件

    if epoch == 1000:
        cnn.save_weights('models/jiandu/1000cnn.h5')
    if epoch == 2000:
        cnn.save_weights('models/jiandu/2000cnn.h5')
cnn.save_weights('models/jiandu/cnn.h5')