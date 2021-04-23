#带危险品的用一个aae重构，不带危险品的用另一个aae重构，重构数据比源数据多十倍
#不带东西截取最大值的后9个和包括自己前11个
#不带东西截取最大值后一个的后一个开始的后8个和不包括自己前10个（最大值和后一个留下）
#带东西截取最大值的第前20个值的前四个和包括自己后六个
#lin取150保证抽取中间有效数据，减少边缘不确定数据
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
cut1=14
cut2=5
lin=150
ww=1
lin2=int((lin*2)/ww)
def read_data_cut1(filenames,kmeans1):
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

        feat = temp_feature - kmeans1
        feat = np.square(feat)
        feat = np.sum(feat, axis=1)
        feat = np.sqrt(feat)
        a = np.argmax(feat)  # 返回feature最大值位置
        idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lin,a-cut1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a+cut1+2,int(temp_feature.shape[0] / 2) + lin, ww)])  # 取中心点处左右分布数据
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
        elif ('ljy' in filename):
            temp_label2 = 1
        elif ('czn' in filename):
            temp_label2 = 2
        elif ('tk' in filename):
            temp_label2 = 3

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
    label2 = np_utils.to_categorical(label2)
    return np.array(feature[:, :270]), np.array(label), np.array(label2)
def read_data_cut2(filenames, kmeans2):
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
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2) - lin,
                                         int(csvdata.shape[0] / 2) + lin, ww)])  # 取中心点处左右分布数据
        temp_feature = csvdata[idx,]

        feat = temp_feature - kmeans2
        feat = np.square(feat)
        feat = np.sum(feat, axis=1)
        feat = np.sqrt(feat)
        a = np.argmax(feat)  # 返回feature最大值位置
        idx1 = np.array([j for j in range(int(temp_feature.shape[0] / 2) - lin, a-20-cut2+1, ww)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range(a-20+cut2+1, int(temp_feature.shape[0] / 2) + lin, ww)])  # 取中心点处左右分布数据
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
        elif ('ljy' in filename):
            temp_label2 = 1
        elif ('czn' in filename):
            temp_label2 = 2
        elif ('tk' in filename):
            temp_label2 = 3

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
        # label = np_utils.to_categorical(label)
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
        elif ('ljy' in filename):
            temp_label2 = 1
        elif ('czn' in filename):
            temp_label2 = 2
        elif ('tk' in filename):
            temp_label2 = 3
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
def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['zb', 'ljy', 'czn']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 30)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]

    trainfile += filenames[:90]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable,domain_label = read_data(trainfile)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    #print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    print(feature)
    k = np.arange(90)
    for i in range(0, 90):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:75]
    # np.random.shuffle(trainfile)

    for name in ['zb', 'ljy', 'czn']:
        for j in ["1M"]:  # "1S", "2S"
            for i in [i for i in range(0, 30)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
    trainfile2 += filenames[:90]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable,domain_label = read_data(trainfile2)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    #print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(90)
    for i in range(0, 90):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:75]
    #np.random.shuffle(trainfile2)

    testfile = trainfile[60:]
    trainfile = trainfile[:75]
    testfile2 = trainfile2[60:]
    trainfile2 = trainfile2[:75]

    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile
def other_file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for j in ["0"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile += filenames[:30]
    filenames = []
    trainfile = np.array(trainfile)
    feature, lable,domain_label = read_data(trainfile)

    kmeans1 = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans1.fit_predict(feature)
    print(kmeans1.cluster_centers_.shape)
    #print(kmeans1.cluster_centers_)
    feature = feature - kmeans1.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(30)
    for i in range(0, 30):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        #print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:25]
    #np.random.shuffle(trainfile)

    # feature, lable, domain_label = read_data(trainfile)
    # feature = feature - kmeans1.cluster_centers_
    # feature = np.square(feature)
    # feature = np.sum(feature, axis=1)
    # feature = np.sqrt(feature)
    # k = np.arange(25)
    # for i in range(0, 25):
    #     k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
    #     print("(不带东西特征)i为", end='')
    #     print(i)
    #     print(feature[i * lin2:(i + 1) * lin2])


    for j in ["1M"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "tk-2.5-M/" + "tk-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile2 += filenames[:30]
    filenames = []
    trainfile2 = np.array(trainfile2)
    feature, lable,domain_label = read_data(trainfile2)

    kmeans2 = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans2.fit_predict(feature)
    print(kmeans2.cluster_centers_.shape)
    #print(kmeans2.cluster_centers_)
    feature = feature - kmeans2.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(30)
    for i in range(0, 30):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        # print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:25]
    #np.random.shuffle(trainfile2)

    # feature, lable, domain_label = read_data(trainfile2)
    # feature = feature - kmeans2.cluster_centers_
    # feature = np.square(feature)
    # feature = np.sum(feature, axis=1)
    # feature = np.sqrt(feature)
    # k = np.arange(25)
    # for i in range(0, 25):
    #     k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
    #     print("(不带东西特征)i为", end='')
    #     print(i)
    #     print(feature[i * lin2:(i + 1) * lin2])


    testfile = trainfile[20:]
    trainfile = trainfile[:25]
    testfile2 = trainfile2[20:]
    trainfile2 = trainfile2[:25]

    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile,kmeans1.cluster_centers_,kmeans2.cluster_centers_
def build_encoder(latent_dim, img_shape):
    deterministic = 1
    img = Input(shape=img_shape)
    h = Flatten()(img)
    h = Dense(512)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dense(512)(h)
    h = LeakyReLU(alpha=0.2)(h)
    latent_repr = Dense(latent_dim)(h)
    return Model(img, latent_repr)
def build_discriminator(latent_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_decoder(latent_dim, img_shape):
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    z = Input(shape=(latent_dim,))
    img = model(z)
    return Model(z, img)
def build_encoder2(latent_dim, img_shape):
    deterministic = 1
    img = Input(shape=img_shape)
    h = Flatten()(img)
    h = Dense(512)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dense(512)(h)
    h = LeakyReLU(alpha=0.2)(h)
    latent_repr = Dense(latent_dim)(h)
    return Model(img, latent_repr)
def build_discriminator2(latent_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_decoder2(latent_dim, img_shape):
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    z = Input(shape=(latent_dim,))
    img = model(z)
    return Model(z, img)
img_rows = 15
img_cols = 18
channels = 1
img_shape = (img_rows, img_cols, channels)
# Results can be found in just_2_rv
# latent_dim = 2
latent_dim = 10

optimizer = Adam(0.0002, 0.5)
optimizer2 = Adam(0.0002, 0.5)
# Build and compile the discriminator
discriminator = build_discriminator(latent_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
discriminator2 = build_discriminator2(latent_dim)
discriminator2.compile(loss='binary_crossentropy', optimizer=optimizer2, metrics=['accuracy'])
# In[27]:


# Build the encoder / decoder
encoder = build_encoder(latent_dim, img_shape)
decoder = build_decoder(latent_dim, img_shape)
encoder2 = build_encoder2(latent_dim, img_shape)
decoder2 = build_decoder2(latent_dim, img_shape)
# In[28]:


# The generator takes the image, encodes it and reconstructs it
# from the encoding
img = Input(shape=img_shape)
encoded_repr = encoder(img)
reconstructed_img = decoder(encoded_repr)
img2 = Input(shape=img_shape)
encoded_repr2 = encoder2(img)
reconstructed_img2 = decoder2(encoded_repr2)
# For the adversarial_autoencoder model we will only train the generator
# It will say something like:
#   UserWarning: Discrepancy between trainable weights and collected trainable weights,
#   did you set `model.trainable` without calling `model.compile` after ?
# We only set trainable to false for the discriminator when it is part of the autoencoder...
discriminator.trainable = False
discriminator2.trainable = False
# The discriminator determines validity of the encoding
validity = discriminator(encoded_repr)
validity2 = discriminator2(encoded_repr2)
# The adversarial_autoencoder model  (stacked generator and discriminator)
adversarial_autoencoder = Model(img, [reconstructed_img, validity])
adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)
adversarial_autoencoder2 = Model(img, [reconstructed_img2, validity2])
adversarial_autoencoder2.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer2)
# In[29]:


discriminator.summary()
discriminator2.summary()
# In[30]:


epochs = 5000
batch_size = 6000
sample_interval = 100


# Rescale -1 to 1
trainfile_array, testfile_array = file_array()#
print(trainfile_array)
print(testfile_array)
train_feature, train_label ,train_domain_label= read_data(trainfile_array)
test_feature, test_label,test_domain_label = read_data(testfile_array)

trainfile_other, testfile_other,kmeans1,kmeans2= other_file_array()#
print(trainfile_other)
print(trainfile_other.shape)
train_feature_ot1, train_label_ot1,train_domain_label_ot = read_data_cut1(trainfile_other[:25],kmeans1)
train_feature_ot2, train_label_ot2,test_domain_label_ot = read_data_cut2(trainfile_other[25:],kmeans2)
train_feature_ot=np.concatenate((train_feature_ot1,train_feature_ot2), axis=0)
print(train_feature_ot)
train_label_ot=np.concatenate((train_label_ot1,train_label_ot2), axis=0)
train_label_ot = np_utils.to_categorical(train_label_ot)
print(train_label_ot.shape)
#全局归化为-1~1
a=np.concatenate((train_feature, train_feature_ot), axis=0)
train_feature = ((train_feature.astype('float32')-np.min(a))-(np.max(a)-np.min(a))/2.0)/((np.max(a)-np.min(a))/2)
test_feature = ((test_feature.astype('float32')-np.min(test_feature))-(np.max(test_feature)-np.min(test_feature))/2.0)/((np.max(test_feature)-np.min(test_feature))/2)
train_feature_ot = ((train_feature_ot.astype('float32')-np.min(a))-(np.max(a)-np.min(a))/2.0)/((np.max(a)-np.min(a))/2)
#test_feature_ot = ((test_feature_ot.astype('float32')-np.min(test_feature_ot))-(np.max(test_feature_ot)-np.min(test_feature_ot))/2.0)/((np.max(test_feature_ot)-np.min(test_feature_ot))/2)

print(train_feature)
print(test_feature)
X_train1 =train_feature[:75*lin2]
print(X_train1.shape)
X_test1 =test_feature[:5*lin2]
print(X_test1.shape)
X_train1 = X_train1.reshape([X_train1.shape[0], img_rows, img_cols])
X_test1 = X_test1.reshape([X_test1.shape[0], img_rows, img_cols])
X_train1 = np.expand_dims(X_train1, axis=3)
X_test1 = np.expand_dims(X_test1, axis=3)

X_train2 =train_feature[75*lin2:]
print(X_train2.shape)
X_test2 =test_feature[5*lin2:]
print(X_test2.shape)
X_train2 = X_train2.reshape([X_train2.shape[0], img_rows, img_cols])
X_test2 = X_test2.reshape([X_test2.shape[0], img_rows, img_cols])
X_train2 = np.expand_dims(X_train2, axis=3)
X_test2 = np.expand_dims(X_test2, axis=3)

train_feature_ot = train_feature_ot.reshape([train_feature_ot.shape[0], img_rows, img_cols])
#test_feature_ot = test_feature_ot.reshape([test_feature_ot.shape[0], img_rows, img_cols])
train_feature_ot = np.expand_dims(train_feature_ot, axis=3)
#test_feature_ot = np.expand_dims(test_feature_ot, axis=3)
print("train_feature_ot")
print(train_feature_ot.shape)
# Adversarial ground truths
valid1 = np.ones((batch_size, 1))
fake1 = np.zeros((batch_size, 1))
valid2 = np.ones((batch_size, 1))
fake2 = np.zeros((batch_size, 1))


def sample_prior(latent_dim, batch_size):
    return np.random.normal(size=(batch_size, latent_dim))

discriminator.load_weights('models/aae-csi2/discriminator.h5')
discriminator2.load_weights('models/aae-csi2/discriminator2.h5')
encoder.load_weights('models/aae-csi2/encoder.h5')
encoder2.load_weights('models/aae-csi2/encoder2.h5')
adversarial_autoencoder.load_weights('models/aae-csi2/adversarial_autoencoder.h5')
adversarial_autoencoder2.load_weights('models/aae-csi2/adversarial_autoencoder2.h5')


train_mid1 = encoder.predict(X_train1)
train_mid2 = encoder2.predict(X_train2)

data=sample_prior(latent_dim, 3*25*lin2)
scdata1=decoder.predict(data)
scdata2=decoder2.predict(data)

X_SCdata1=0.5*X_train1+0.5*scdata1
X_SCdata2=0.5*X_train2+0.5*scdata2
X_SCdata1_label=train_label[:lin2*75]
X_SCdata2_label=train_label[lin2*75:]
X_SCdata1_domain_label=train_domain_label[:3*25*lin2]
X_SCdata2_domain_label=train_domain_label[3*25*lin2:]

X_SCdata=np.concatenate((X_SCdata1,X_SCdata2), axis=0)
X_SCdata_label=np.concatenate((X_SCdata1_label,X_SCdata2_label), axis=0)
X_SCdata_domain_label=np.concatenate((X_SCdata1_domain_label,X_SCdata2_domain_label), axis=0)
all_data=np.concatenate((X_SCdata1,X_SCdata2), axis=0)
print(all_data.shape)
all_data=np.concatenate((all_data,train_feature_ot), axis=0)
all_data=np.concatenate((all_data,train_feature_ot), axis=0)
all_data=np.concatenate((all_data,train_feature_ot), axis=0)
print(all_data.shape)

latent_dim = 64
latent_dim2=128

def build_ed(latent_dim2, img_shape):
    deterministic = 1
    img = Input(shape=img_shape)
    h = Flatten()(img)
    h = Dense(1000)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dense(1000)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Dense(1000)(h)
    h = LeakyReLU(alpha=0.2)(h)
    # h = Dense(512)(h)
    # h = LeakyReLU(alpha=0.2)(h)
    # h = Dense(512)(h)
    # h = LeakyReLU(alpha=0.2)(h)
    # h = Dense(512)(h)
    # h = LeakyReLU(alpha=0.2)(h)
    latent_repr = Dense(latent_dim2)(h)
    return Model(img, latent_repr)
def build_class(latent_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_dis(latent_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(3, activation="softmax"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)
def build_dd(latent_dim2, img_shape):
    model = Sequential()
    model.add(Dense(1000, input_dim=latent_dim2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1000))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1000))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(512, input_dim=latent_dim2))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
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

classer.load_weights('models/fido3_lat10-64upclasser2/classer.h5')
ed.load_weights('models/fido3_lat10-64upclasser2/ed.h5')
dd.load_weights('models/fido3_lat10-64upclasser2/dd.h5')
dis.load_weights('models/fido3_lat10-64upclasser2/dis.h5')
dis_model.load_weights('models/fido3_lat10-64upclasser2/dis_model.h5')
class_model.load_weights('models/fido3_lat10-64upclasser2/class_model.h5')
sc_fido.load_weights('models/fido3_lat10-64upclasser2/sc_fido.h5')


non_mid=ed.predict(X_train1)
non_mid=non_mid[:,:latent_dim]
non_pre=classer.predict(non_mid)
yes_mid=ed.predict(X_train2)
yes_mid=yes_mid[:,:latent_dim]
yes_pre=classer.predict(yes_mid)
print(non_mid)
print(non_mid.shape)
print(yes_mid)
print(yes_mid.shape)
print(non_pre)
print(non_pre.shape)
print(yes_pre)
print(yes_pre.shape)

a1=[0,0]
a2=[0,0]
k1=[0,0]
non_pre_1 = np.arange(len(non_pre))
for i in range(0,int(len(non_pre))):
    if non_pre[i][0]>=non_pre[i][1]:
        a1[0]=a1[0]+1
        non_pre_1[i] =1
    if non_pre[i][0] < non_pre[i][1]:
        a1[1] = a1[1] + 1
        non_pre_1[i] = 0

acc_non_pre=float(a1[0])/float(len(non_pre))
a1=[0,0]
for i in range(0,int(len(non_pre_1))):
    if non_pre_1[i]==1:
        k1[0]=k1[0]+1
        a1[0] = a1[0] + 1
    if non_pre_1[i] == 0:
        k1[1] = k1[1] + 1
        a1[1] = a1[1] + 1
    if (k1[0]+k1[1]==lin2):
        if k1[0]>=k1[1]:
            a2[0]=a2[0]+1
        if k1[0]<k1[1]:
            a2[1]=a2[1]+1
        k1=[0,0]
acc_non_pre_vot=float(a2[0])/float(len(non_pre_1)/lin2)
print(a1)
print(a2)



print("不带东西源标签数据准确率：")
print(acc_non_pre)
print("投票后不带东西源标签数据准确率：")
print(acc_non_pre_vot)


a1=[0,0]
a2=[0,0]
k1=[0,0]
for i in range(0,int(len(yes_pre))):
    if yes_pre[i][0]>yes_pre[i][1]:a1[0]=a1[0]+1
    if yes_pre[i][0] <= yes_pre[i][1]: a1[1] = a1[1] + 1
# print("a1")
print(a1)
# acc_yes_pre=float(a1[1])/float(len(yes_pre))
a1=[0,0]
yes_pre_1 = np.arange(len(yes_pre))
for i in range(0,int(len(yes_pre))):
    if yes_pre[i][0]>yes_pre[i][1]:
        a1[0]=a1[0]+1
        yes_pre_1[i] =1
    if yes_pre[i][0] <= yes_pre[i][1]:
        a1[1] = a1[1] + 1
        yes_pre_1[i] = 0

acc_yes_pre=float(a1[1])/float(len(yes_pre))
a1=[0,0]
for i in range(0,int(len(yes_pre_1))):
    if yes_pre_1[i]==1:
        k1[0]=k1[0]+1
        a1[0] = a1[0] + 1
    if yes_pre_1[i] == 0:
        k1[1] = k1[1] + 1
        a1[1] = a1[1] + 1
    if (k1[0]+k1[1]==lin2):
        if k1[0]>k1[1]:
            a2[0]=a2[0]+1
        if k1[0]<=k1[1]:
            a2[1]=a2[1]+1
        k1=[0,0]
acc_yes_pre_vot=float(a2[1])/float(len(yes_pre_1)/lin2)
print(a1)
print(a2)
print("带东西源标签数据准确率：")
print(acc_yes_pre)
print("投票后带东西源标签数据准确率：")
print(acc_yes_pre_vot)


non_mid=ed.predict(X_SCdata1[:75*lin2])
non_mid=non_mid[:,:latent_dim]
non_pre=classer.predict(non_mid)
yes_mid=ed.predict(X_SCdata2[:75*lin2])
yes_mid=yes_mid[:,:latent_dim]
yes_pre=classer.predict(yes_mid)
print(non_mid)
print(non_mid.shape)
print(yes_mid)
print(yes_mid.shape)
print(non_pre)
print(non_pre.shape)
print(yes_pre)
print(yes_pre.shape)

a1=[0,0]
a2=[0,0]
k1=[0,0]
non_pre_1 = np.arange(len(non_pre))
for i in range(0,int(len(non_pre))):
    if non_pre[i][0]>=non_pre[i][1]:
        a1[0]=a1[0]+1
        non_pre_1[i] =1
    if non_pre[i][0] < non_pre[i][1]:
        a1[1] = a1[1] + 1
        non_pre_1[i] = 0

acc_non_pre=float(a1[0])/float(len(non_pre))
a1=[0,0]
for i in range(0,int(len(non_pre_1))):
    if non_pre_1[i]==1:
        k1[0]=k1[0]+1
        a1[0] = a1[0] + 1
    if non_pre_1[i] == 0:
        k1[1] = k1[1] + 1
        a1[1] = a1[1] + 1
    if (k1[0]+k1[1]==lin2):
        if k1[0]>=k1[1]:
            a2[0]=a2[0]+1
        if k1[0]<k1[1]:
            a2[1]=a2[1]+1
        k1=[0,0]
acc_non_pre_vot=float(a2[0])/float(len(non_pre_1)/lin2)
print(a1)
print(a2)



print("合成的不带东西源标签数据准确率：")
print(acc_non_pre)
print("投票后合成的不带东西源标签数据准确率：")
print(acc_non_pre_vot)


a1=[0,0]
a2=[0,0]
k1=[0,0]
for i in range(0,int(len(yes_pre))):
    if yes_pre[i][0]>yes_pre[i][1]:a1[0]=a1[0]+1
    if yes_pre[i][0] <= yes_pre[i][1]: a1[1] = a1[1] + 1
# print("a1")
print(a1)
# acc_yes_pre=float(a1[1])/float(len(yes_pre))
a1=[0,0]
yes_pre_1 = np.arange(len(yes_pre))
for i in range(0,int(len(yes_pre))):
    if yes_pre[i][0]>yes_pre[i][1]:
        a1[0]=a1[0]+1
        yes_pre_1[i] =1
    if yes_pre[i][0] <= yes_pre[i][1]:
        a1[1] = a1[1] + 1
        yes_pre_1[i] = 0

acc_yes_pre=float(a1[1])/float(len(yes_pre))
a1=[0,0]
for i in range(0,int(len(yes_pre_1))):
    if yes_pre_1[i]==1:
        k1[0]=k1[0]+1
        a1[0] = a1[0] + 1
    if yes_pre_1[i] == 0:
        k1[1] = k1[1] + 1
        a1[1] = a1[1] + 1
    if (k1[0]+k1[1]==lin2):
        if k1[0]>k1[1]:
            a2[0]=a2[0]+1
        if k1[0]<=k1[1]:
            a2[1]=a2[1]+1
        k1=[0,0]
acc_yes_pre_vot=float(a2[1])/float(len(yes_pre_1)/lin2)
print(a1)
print(a2)
print("合成的带东西源标签数据准确率：")
print(acc_yes_pre)
print("投票后合成的带东西源标签数据准确率：")
print(acc_yes_pre_vot)


non_mid3=ed.predict(train_feature_ot[:(lin2-cut1*2-2)*25])
non_mid3=non_mid3[:,:latent_dim]
non_pre3=classer.predict(non_mid3)
yes_mid3=ed.predict(train_feature_ot[(lin2-cut1*2-2)*25:])
yes_mid3=yes_mid3[:,:latent_dim]
yes_pre3=classer.predict(yes_mid3)
print(non_mid3)
print(non_mid3.shape)
print(yes_mid3)
print(yes_mid3.shape)
print(non_pre3)
print(non_pre3.shape)
print(yes_pre3)
print(yes_pre3.shape)

a1=[0,0]
a2=[0,0]
k1=[0,0]
non_pre3_1 = np.arange(len(non_pre3))
for i in range(0,int(len(non_pre3))):
    if non_pre3[i][0]>=non_pre3[i][1]:
        a1[0]=a1[0]+1
        non_pre3_1[i] =1
    if non_pre3[i][0] < non_pre3[i][1]:
        a1[1] = a1[1] + 1
        non_pre3_1[i] = 0

# for i in range(0,int(len(non_pre3_1)/(lin2-cut1*2-2))):
#     print("(不带东西)i为", end='')
#     print(i)
#     print(non_pre3_1[i * (lin2-cut1*2-2):(i + 1) * (lin2-cut1*2-2)])

acc_non_pre3=float(a1[0])/float(len(non_pre3))
a1=[0,0]
for i in range(0,int(len(non_pre3_1))):
    if non_pre3_1[i]==1:
        k1[0]=k1[0]+1
        a1[0] = a1[0] + 1
    if non_pre3_1[i] == 0:
        k1[1] = k1[1] + 1
        a1[1] = a1[1] + 1
    if (k1[0]+k1[1]==(lin2-cut1*2-2)):
        if k1[0]>=k1[1]:
            a2[0]=a2[0]+1
        if k1[0]<k1[1]:
            a2[1]=a2[1]+1
        k1=[0,0]
acc_non_pre3_vot=float(a2[0])/float(len(non_pre3_1)/(lin2-cut1*2-2))
print(a1)
print(a2)

print("无标签不带东西数据准确率：")
print(acc_non_pre3)
print("投票后无标签不带东西数据准确率：")
print(acc_non_pre3_vot)


a1=[0,0]
a2=[0,0]
k1=[0,0]
for i in range(0,int(len(yes_pre3))):
    if yes_pre3[i][0]>yes_pre3[i][1]:a1[0]=a1[0]+1
    if yes_pre3[i][0] <= yes_pre3[i][1]: a1[1] = a1[1] + 1
# print("a1")
print(a1)
# acc_yes_pre=float(a1[1])/float(len(yes_pre))
a1=[0,0]
yes_pre3_1 = np.arange(len(yes_pre3))
for i in range(0,int(len(yes_pre3))):
    if yes_pre3[i][0]>yes_pre3[i][1]:
        a1[0]=a1[0]+1
        yes_pre3_1[i] =1
    if yes_pre3[i][0] <= yes_pre3[i][1]:
        a1[1] = a1[1] + 1
        yes_pre3_1[i] = 0

# for i in range(0,int(len(yes_pre3_1)/(lin2-cut2*2))):
#     print("(带东西)i为", end='')
#     print(i)
#     print(yes_pre3_1[i * (lin2-cut2*2):(i + 1) * (lin2-cut2*2)])

acc_yes_pre3=float(a1[1])/float(len(yes_pre3))
a1=[0,0]
for i in range(0,int(len(yes_pre3_1))):
    if yes_pre3_1[i]==1:
        k1[0]=k1[0]+1
        a1[0] = a1[0] + 1
    if yes_pre3_1[i] == 0:
        k1[1] = k1[1] + 1
        a1[1] = a1[1] + 1
    if (k1[0]+k1[1]==(lin2-cut2*2)):
        if k1[0]>k1[1]:
            a2[0]=a2[0]+1
        if k1[0]<=k1[1]:
            a2[1]=a2[1]+1
        k1=[0,0]
acc_yes_pre3_vot=float(a2[1])/float(len(yes_pre3_1)/(lin2-cut2*2))
print(a1)
print(a2)
print("无标签带东西数据准确率：")
print(acc_yes_pre3)
print("投票后无标签带东西数据准确率：")
print(acc_yes_pre3_vot)