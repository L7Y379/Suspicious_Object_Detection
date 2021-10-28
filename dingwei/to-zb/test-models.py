import pandas as pd
import os
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
        print(train_feature.shape)
        train_label = csvdata[:, 270:]
        print(train_label.shape)
        temp_label = np.tile(0, (train_label.shape[0],))
        print("temp_label" + str(temp_label.shape))
        print(temp_label)

        fn = filepath + name + "/test" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        test_feature = csvdata[:, 0:270]
        print(test_feature.shape)
        test_label = csvdata[:, 270:]
        print(test_label.shape)



    for name in ['ori_tk']:
        fn = filepath + name + "/train" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        train_feature = np.concatenate((train_feature, csvdata[:, 0:270]), axis=0)
        print(train_feature.shape)
        train_label = np.concatenate((train_label, csvdata[:, 270:]), axis=0)
        print(train_label.shape)
        temp_label2 = np.tile(1, (2400,))
        temp_label= np.concatenate((temp_label, temp_label2), axis=0)
        domain_label = np_utils.to_categorical(temp_label)
        print("temp_label" + str(temp_label.shape))
        print(temp_label)
        print("domain_label"+str(domain_label.shape))
        print(domain_label[2400:])

        fn = filepath + name + "/test" + filetype
        if os.path.exists(fn) == False:
            print(fn + " doesn't exit.")
            exit(1)
        csvdata = pd.read_csv(fn, header=None)
        csvdata = np.array(csvdata, dtype=np.float64)
        test_feature = np.concatenate((test_feature, csvdata[:, 0:270]), axis=0)
        print(test_feature.shape)
        test_label = np.concatenate((test_label, csvdata[:, 270:]), axis=0)
        print(test_label.shape)
    return np.array(train_feature), np.array(train_label),np.array(domain_label),np.array(test_feature),np.array(test_label)
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
        fn = filepath + name+ filetype
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


    return np.array(train_feature), np.array(train_label), np.array(domain_label),np.array(test_feature),np.array(test_label)
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
        train_label = csvdata[:, 270:]
        print(train_label.shape)

    return np.array(train_feature), np.array(train_label)
train_feature, train_label,train_domain_label,test_feature,test_label = read_data()
train_feature_ot, train_label_ot = read_data_ot()
print("train_feature"+str(train_feature.shape))
print("train_label"+str(train_label.shape))
print("test_feature"+str(test_feature.shape))
print("test_label"+str(test_label.shape))
print("domain_label"+str(train_domain_label.shape))
print("train_feature_ot"+str(train_feature_ot.shape))
print("train_label_ot"+str(train_label_ot.shape))
img_rows = 15
img_cols = 18
channels = 1
img_shape = (img_rows, img_cols, channels)

epochs = 5000
batch_size = 4800

train_feature = train_feature.reshape([train_feature.shape[0], img_rows, img_cols])
train_feature = np.expand_dims(train_feature, axis=3)
test_feature = test_feature.reshape([test_feature.shape[0], img_rows, img_cols])
test_feature = np.expand_dims(test_feature, axis=3)
train_feature_ot = train_feature_ot.reshape([train_feature_ot.shape[0], img_rows, img_cols])
train_feature_ot = np.expand_dims(train_feature_ot, axis=3)



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
    model.add(Dense(2, activation="relu"))
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
    model.add(Dense(800, input_dim=latent_dim2,activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(800, activation="relu"))
    model.add(Dense(np.prod(img_shape), activation='relu'))
    model.add(Reshape(img_shape))
    z = Input(shape=(latent_dim2,))
    img = model(z)
    return Model(z, img)

opt = Adam(0.0002, 0.5)
classer = build_class(latent_dim)
classer.compile(loss='mse', optimizer=opt)
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
class_model.compile(loss='mse', optimizer=opt)
dis_model=Model(img3,validity2)
dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



classer.load_weights('models/3000classer.h5')
ed.load_weights('models/3000ed.h5')
dd.load_weights('models/3000dd.h5')
dis.load_weights('models/3000dis.h5')

mid = ed.predict(train_feature)
mid = mid[:, :latent_dim]
pre = classer.predict(mid)
a=pre-train_label
a=np.power(a, 2)
a=np.sum(a, axis=1)
a = np.sqrt(a)
#print(a.shape)
a=np.mean(a)
print('训练数据精度：'+str(a))

domain_mid = ed.predict(train_feature[:2400])
domain_mid = domain_mid[:, latent_dim:]
pre = dis.predict(domain_mid)
print(pre)
a1 = [0, 0]
a2 = [0, 0]
k1 = [0, 0]
non_pre4_1 = np.arange(len(pre))
for i in range(0, int(len(pre))):
    if pre[i][0] >= pre[i][1]:
        a1[0] = a1[0] + 1
        non_pre4_1[i] = 1
    if pre[i][0] < pre[i][1]:
        a1[1] = a1[1] + 1
        non_pre4_1[i] = 0

acc = float(a1[0]) / float(len(pre))
print(a1)
print('训练数据域分类精度：'+str(acc))

domain_mid = ed.predict(train_feature[2400:])
domain_mid = domain_mid[:, latent_dim:]
pre = dis.predict(domain_mid)
print(pre)
a1 = [0, 0]
a2 = [0, 0]
k1 = [0, 0]
non_pre4_1 = np.arange(len(pre))
for i in range(0, int(len(pre))):
    if pre[i][0] > pre[i][1]:
        a1[0] = a1[0] + 1
        non_pre4_1[i] = 1
    if pre[i][0] <= pre[i][1]:
        a1[1] = a1[1] + 1
        non_pre4_1[i] = 0

acc = float(a1[1]) / float(len(pre))
print(a1)
print('训练数据域分类精度：'+str(acc))

mid1 = ed.predict(test_feature)
mid1 = mid1[:, :latent_dim]
pre1 = classer.predict(mid1)
a = pre1 - test_label
a = np.power(a, 2)
a = np.sum(a, axis=1)
a = np.sqrt(a)
#print(a.shape)
a = np.mean(a)
print('测试数据精度：' + str(a))
# pre1_T=pre1.T
# print(pre1_T)
# test_label_T = test_label.T
# print(test_label_T)
# b=mean_squared_error(pre1_T, test_label_T,multioutput='raw_values')
# print(b.shape)
# print(b)
# b=np.sqrt(mean_squared_error(pre1_T, test_label_T,multioutput='raw_values'))
# print(b.shape)
# print(b)
# print(np.mean(b))
domain_mid = ed.predict(test_feature[:1200])
domain_mid = domain_mid[:, latent_dim:]
pre = dis.predict(domain_mid)
print(pre)
a1 = [0, 0]
a2 = [0, 0]
k1 = [0, 0]
non_pre4_1 = np.arange(len(pre))
for i in range(0, int(len(pre))):
    if pre[i][0] >= pre[i][1]:
        a1[0] = a1[0] + 1
        non_pre4_1[i] = 1
    if pre[i][0] < pre[i][1]:
        a1[1] = a1[1] + 1
        non_pre4_1[i] = 0

acc = float(a1[0]) / float(len(pre))
print(a1)
print('训练数据域分类精度：'+str(acc))

domain_mid = ed.predict(test_feature[1200:])
domain_mid = domain_mid[:, latent_dim:]
pre = dis.predict(domain_mid)
print(pre)
a1 = [0, 0]
a2 = [0, 0]
k1 = [0, 0]
non_pre4_1 = np.arange(len(pre))
for i in range(0, int(len(pre))):
    if pre[i][0] > pre[i][1]:
        a1[0] = a1[0] + 1
        non_pre4_1[i] = 1
    if pre[i][0] <= pre[i][1]:
        a1[1] = a1[1] + 1
        non_pre4_1[i] = 0

acc = float(a1[1]) / float(len(pre))
print(a1)
print('训练数据域分类精度：'+str(acc))


mid1 = ed.predict(train_feature_ot)
mid1 = mid1[:, :latent_dim]
pre1 = classer.predict(mid1)
a = pre1 - train_label_ot
a = np.power(a, 2)
a = np.sum(a, axis=1)
a = np.sqrt(a)
#print(a.shape)
a = np.mean(a)
print('他人数据精度：' + str(a))














