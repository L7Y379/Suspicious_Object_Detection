#!/usr/bin/env python
# coding: utf-8

# # Code explanation
#
# At [this blog](http://nnormandin.com/science/2017/07/01/cvae.html) quite a few details of typical Keras models are explained. Note that older Keras versions had different ways to handle merging of layers as for a variational autoencoder (see e.g. [here](https://github.com/keras-team/keras/issues/3921)).
#
# However, I don't get why there is a `sample_z` function. The purpose of an adversarial autoencoder is that it would not need differentiable probability densities in the latent layer, or that's what I thought. The latent representation should be compared to samples from a normal distribution by the discriminator.
#
# Ah, that is actually in the original paper! The authors distinguish three different autoencoders. (1) The deterministic autoencoder (that's if you skip the layer containing random variables altogether). (2) An autoencoder that uses a Gaussian posterior. In this case we can indeed use the same renormalization trick as in Kingma and Welling. (3) A general autoencoder with a "univeral approximate posterior" where we add noise to the input of the encoder.
#
# The network has to match q(z) to p(z) by only exploiting the stochasticity in the data distribution in the deterministic case. However, the authors found that for all different types an extensive sweep over hyperparameters did obtain similiar test-likelihoods. All their reported results where subsequently with a deterministic autoencoder.

# In[1]:
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
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

lin=60
ww=1
lin2=int((lin*2)/ww)
def read_data(filenames):
    i = 0
    feature = []
    label = []
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
        if ('-0-' in filename):
            temp_label = 0
        elif ('-1M-' in filename):
            temp_label = 1
        elif ('2M' in filename):
            temp_label = 2
        elif ('-3M-' in filename):
            temp_label = 3
        temp_label = np.tile(temp_label, (temp_feature.shape[0],))
        #temp_label = tf.Session().run(tf.one_hot(temp_label, N_CLASS))
        if i == 0:
            feature = temp_feature
            label = temp_label
            i = i + 1
        else:
            feature = np.concatenate((feature, temp_feature), axis=0)  # 拼接

    return np.array(feature[:, :270]), np.array(feature[:, 270:])
def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for j in ["0"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "zb-2.5-M/" + "zb-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile += filenames[:30]
    filenames = []
    trainfile =np.array(trainfile)
    feature,lable=read_data(trainfile)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(30)
    for i in range(0, 30):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        print(k[i])
    trainfile = trainfile[np.argsort(k)]
    trainfile = trainfile[:25]
    np.random.shuffle(trainfile)

    for j in ["1M"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "zb-2.5-M/" + "zb-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
    trainfile2 += filenames[:30]
    filenames = []
    trainfile2 =np.array(trainfile2)
    feature,lable=read_data(trainfile2)

    kmeans = KMeans(n_clusters=1, n_init=50)
    pred_train = kmeans.fit_predict(feature)
    print(kmeans.cluster_centers_.shape)
    print(kmeans.cluster_centers_)
    feature = feature - kmeans.cluster_centers_
    feature = np.square(feature)
    feature = np.sum(feature, axis=1)
    feature = np.sqrt(feature)
    k = np.arange(30)
    for i in range(0, 30):
        k[i] = np.mean(feature[i * lin2:(i + 1) * lin2])
        print(k[i])
    trainfile2 = trainfile2[np.argsort(k)]
    trainfile2 = trainfile2[:25]
    np.random.shuffle(trainfile2)


    testfile = trainfile[20:]
    trainfile = trainfile[:20]
    testfile2 = trainfile2[20:]
    trainfile2 = trainfile2[:20]

    trainfile=np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
    return trainfile, testfile


# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    m, n = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  #
        centroids[i, :] = dataSet[index, :]
    return centroids

# k均值聚类
def KMeans1(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i])

    plt.show()
# In the case of a non-deterministic autoencoder we have a layer with random variables where we sample from using the renormalization trick. See my website on [inference](https://www.annevanrossum.com/blog/2018/01/30/inference-in-deep-learning/) and other [variance reduction methods](https://www.annevanrossum.com/blog/2018/05/26/random-gradients/).

# In[22]:



# The encoder, discriminator, and decoder have layers of size 512 or 256 and are densely connected. I have not experimented much with the number of nodes. Regarding the activation function leaky rectifiers are used. A rectifier is a function of the form: $f(x) = \max(0,x)$, in other words, making sure the values don't go below zero, but not bounding it from above. The leaky rectifiers are defined through $f(x) = x$ for $x > 0$ and $f(x) = x \alpha$ otherwise. This makes it less likely to have them "stuck" when all there inputs become negative.

# In[23]:


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


# In[24]:


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


# In[25]:


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


# The input are 28x28 images. The optimization used is Adam. The loss is binary cross-entropy.

# In[26]:


img_rows = 15
img_cols = 18
channels = 1
img_shape = (img_rows, img_cols, channels)
# Results can be found in just_2_rv
# latent_dim = 2
latent_dim = 2

optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator(latent_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# In[27]:


# Build the encoder / decoder
encoder = build_encoder(latent_dim, img_shape)
decoder = build_decoder(latent_dim, img_shape)

# In[28]:


# The generator takes the image, encodes it and reconstructs it
# from the encoding
img = Input(shape=img_shape)
encoded_repr = encoder(img)
reconstructed_img = decoder(encoded_repr)

# For the adversarial_autoencoder model we will only train the generator
# It will say something like:
#   UserWarning: Discrepancy between trainable weights and collected trainable weights,
#   did you set `model.trainable` without calling `model.compile` after ?
# We only set trainable to false for the discriminator when it is part of the autoencoder...
discriminator.trainable = False

# The discriminator determines validity of the encoding
validity = discriminator(encoded_repr)

# The adversarial_autoencoder model  (stacked generator and discriminator)
adversarial_autoencoder = Model(img, [reconstructed_img, validity])
adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)

# In[29]:


discriminator.summary()

# In[30]:


epochs = 500
batch_size = 128
sample_interval = 100


# Rescale -1 to 1
trainfile_array, testfile_array = file_array()#
print(trainfile_array)
print(testfile_array)
train_feature, train_label = read_data(trainfile_array)
test_feature, test_label = read_data(testfile_array)

#全局归化为-1~1
# train_feature = train_feature.astype('float32')/np.max(train_feature)
# test_feature = test_feature.astype('float32')/np.max(test_feature)
# tk_feature=tk_feature.astype('float32')/np.max(tk_feature)
train_feature = ((train_feature.astype('float32')-np.min(train_feature))-(np.max(train_feature)-np.min(train_feature))/2.0)/((np.max(train_feature)-np.min(train_feature))/2)
test_feature = ((test_feature.astype('float32')-np.min(test_feature))-(np.max(test_feature)-np.min(test_feature))/2.0)/((np.max(test_feature)-np.min(test_feature))/2)

print(train_feature)
print(test_feature)
#列归化为-1~1
# min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1,1),copy = 1)
# train_feature = min_max_scaler.fit_transform(train_feature)
# test_feature = min_max_scaler.fit_transform(test_feature)
# tk_feature = min_max_scaler.fit_transform(tk_feature)

# #行归化为-1~1
# min_max_scaler1 = preprocessing.MinMaxScaler(feature_range = (-1,1),copy = 1)
# train_feature=train_feature.T
# test_feature=test_feature.T
# tk_feature=tk_feature.T
# train_feature = min_max_scaler1.fit_transform(train_feature)
# test_feature = min_max_scaler1.fit_transform(test_feature)
# tk_feature = min_max_scaler1.fit_transform(tk_feature)
# train_feature=train_feature.T
# test_feature=test_feature.T
# tk_feature=tk_feature.T
X_train =train_feature[20*lin2:]
X_test =test_feature
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape([X_train.shape[0], img_rows, img_cols])
X_test = X_test.reshape([X_test.shape[0], img_rows, img_cols])
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))


# In[ ]:


def sample_prior(latent_dim, batch_size):
    return np.random.normal(size=(batch_size, latent_dim))


# In[31]:


def sample_images(latent_dim, decoder, epoch):
    r, c = 5, 5

    z = sample_prior(latent_dim, r * c)
    gen_imgs = decoder.predict(z)

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/aae-csi/mnist_%d.png" % epoch)
    plt.close()


# # Training
#
# Each epoch a batch is chosen from the images at random. The typical batch size is 128 items out of 60.000 images. The change to pick the same image is minimal (but not zero).
#
# The "real" latent variables for the encoder will be Normal distributed. They all have the same N(0,1) distribution, mu=0, sigma=1. The variables are returned in a 128x10 matrix if we use 10 latent variables.
#
# The discriminator doesn't know that there is order to the "real" and "fake" samples. We can just first train it on all the real ones and then all the fake ones. I don't know if it matters for the training, but we might try to actually build one data structure where this is randomized...
#
#

# In[32]:


for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    latent_fake = encoder.predict(imgs)

    # Here we generate the "TRUE" samples
    latent_real = sample_prior(latent_dim, batch_size)

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(latent_real, valid)
    d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    # Train the generator
    g_loss = adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

    # Plot the progress (every 10th epoch)
    if epoch % 10 == 0:
        print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
        epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))

    # Save generated images (every sample interval, e.g. every 100th epoch)
    if epoch % sample_interval == 0:
        sample_images(latent_dim, decoder, epoch)
adversarial_autoencoder.save_weights('models/aae-csi/aae-csi.h5')
train_mid = encoder.predict(X_train)
test_mid =encoder.predict(X_test)
print(train_mid)
print(test_mid)



k = 2
centroids, clusterAssment = KMeans1(train_mid, k)
showCluster(train_mid, k, centroids, clusterAssment)


m, n = train_mid.shape
for i in range(0,int(m/4)):
    plt.plot(train_mid[i, 0], train_mid[i, 1], 'ob')
for i in range(int(m/4),int(m/2)):
    plt.plot(train_mid[i, 0], train_mid[i, 1], 'or')
for i in range(int(m/2),int(3*m/4)):
    plt.plot(train_mid[i, 0], train_mid[i, 1], 'og')
for i in range(int(3*m/4),m):
    plt.plot(train_mid[i, 0], train_mid[i, 1], 'oy')
plt.show()

