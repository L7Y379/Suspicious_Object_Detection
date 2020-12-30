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
from sklearn.cluster import KMeans
import pandas as pd
import os
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
def file_array():#训练和测试文件名数组
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    testfile = []
    for j in ["0", "3M"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "zb-2.5-M/" + "zb-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
        np.random.shuffle(filenames)
        trainfile += filenames[:20]
        testfile += filenames[20:]
        filenames = []
    trainfile = np.array(trainfile)#20*2
    testfile = np.array(testfile)#10*2
    #print(testfile);
    return trainfile, testfile
def file_array_other():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    for j in ["0"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "czn-2.5-M/" + "czn-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
        np.random.shuffle(filenames)
    filenames = np.array(filenames)#20*2
    return filenames

def read_data(filenames):#读取文件中数据，并贴上标签
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
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2)-120 ,
                                         int(csvdata.shape[0] / 2) +120, 2)])#取中心点处左右分布数据
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
            #label = np.concatenate((label, temp_label), axis=0)
    #data = np.concatenate((feature, label), axis=1)
    #np.random.shuffle(feature)
    return np.array(feature[:, :270]), np.array(feature[:, 270:])
    #return np.array(feature[:, 134:136]), np.array(feature[:, 134:136])

# In the case of a non-deterministic autoencoder we have a layer with random variables where we sample from using the renormalization trick. See my website on [inference](https://www.annevanrossum.com/blog/2018/01/30/inference-in-deep-learning/) and other [variance reduction methods](https://www.annevanrossum.com/blog/2018/05/26/random-gradients/).

# In[22]:


def sample_z(args):
    mu, log_var = args
    batch = K.shape(mu)[0]
    eps = K.random_normal(shape=(batch, latent_dim), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * eps


# The encoder, discriminator, and decoder have layers of size 512 or 256 and are densely connected. I have not experimented much with the number of nodes. Regarding the activation function leaky rectifiers are used. A rectifier is a function of the form: $f(x) = \max(0,x)$, in other words, making sure the values don't go below zero, but not bounding it from above. The leaky rectifiers are defined through $f(x) = x$ for $x > 0$ and $f(x) = x \alpha$ otherwise. This makes it less likely to have them "stuck" when all there inputs become negative.

# In[23]:


def build_encoder(latent_dim, img_shape):
    deterministic = 1
    img = Input(shape=img_shape)
    h = Flatten()(img)
    # h = Dense(512)(h)
    # h = LeakyReLU(alpha=0.2)(h)
    # h = Dense(512)(h)
    # h = LeakyReLU(alpha=0.2)(h)
    if deterministic:
        latent_repr = Dense(latent_dim)(h)
    else:
        mu = Dense(latent_dim)(h)
        log_var = Dense(latent_dim)(h)
        latent_repr = Lambda(sample_z)([mu, log_var])
    return Model(img, latent_repr)


# In[24]:


def build_discriminator(latent_dim):
    model = Sequential()
    # model.add(Dense(512, input_dim=latent_dim))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    encoded_repr = Input(shape=(latent_dim,))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)


# In[25]:


def build_decoder(latent_dim, img_shape):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim,activation='relu'))
    #model.add(LeakyReLU(alpha=0.2))
    # model.add(Dense(512))
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape), activation='sigmoid'))
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
latent_dim = 2
# latent_dim = 8

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


epochs = 2000
batch_size = 100
sample_interval = 100

# Load the dataset
#(X_train, _), (_, _) = mnist.load_data()
trainfile_array, testfile_array = file_array()
tk_files=file_array_other()
train_feature_all, train_label_all = read_data(trainfile_array)
test_feature_all, test_label_all = read_data(testfile_array)
tk_feature,tk_label=read_data(tk_files)

# Rescale -1 to 1
# train_feature_all = train_feature_all.astype('float32')
# test_feature_all = test_feature_all.astype('float32')
X_train = train_feature_all.astype('float32')/ 73.0
X_test = test_feature_all.astype('float32') / 73.0
tk_feature=tk_feature.astype('float32')/73.0
X_train = X_train.reshape([X_train.shape[0], img_rows, img_cols])
X_test = X_test.reshape([X_test.shape[0], img_rows, img_cols])
tk_feature=tk_feature.reshape([tk_feature.shape[0], img_rows, img_cols])
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
tk_feature = np.expand_dims(tk_feature, axis=3)
# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

def sample_prior(latent_dim, batch_size):
    return np.random.normal(size=(batch_size, latent_dim))+0.1


# def sample_images(latent_dim, decoder, epoch):
#     r, c = 5, 5
#
#     z = sample_prior(latent_dim, r * c)
#     gen_imgs = decoder.predict(z)
#
#     fig, axs = plt.subplots(r, c)
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
#             axs[i, j].axis('off')
#             cnt += 1
#     fig.savefig("AAE-CSI-images/CSI_%d.png" % epoch)
#     plt.close()



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
        # k = 2
        # print(latent_fake.shape)
        # centroids, clusterAssment = KMeans1(latent_fake, k)
        # showCluster(latent_fake, k, centroids, clusterAssment)
    # Save generated images (every sample interval, e.g. every 100th epoch)
    # if epoch % sample_interval == 0:
    #     sample_images(latent_dim, decoder, epoch)

train_mid = encoder.predict(X_train)
test_mid =encoder.predict(X_test)
tk_mid = encoder.predict(tk_feature)
print(train_mid)
print(test_mid)

kmeans = KMeans(n_clusters=2,n_init=20).fit(train_mid)
pred_train = kmeans.predict(train_mid)
print(pred_train)
pred_test = kmeans.predict(test_mid)
print(pred_test)
pred_tk=kmeans.predict(tk_mid)

a1=[0,0]
a2=[0,0]
for i in range(0,int(len(pred_train)/2)):
    if pred_train[i]==0:a1[0]=a1[0]+1
    if pred_train[i]==1:a1[1]=a1[1]+1
for j in range(int(len(pred_train)/2),int(len(pred_train))):
    if pred_train[j] == 0: a2[0] = a2[0] + 1
    if pred_train[j] == 1: a2[1] = a2[1] + 1
print(a1)
print(a2)
if((a1[0]+a2[1])>=(a1[1]+a2[0])):
    a=(a1[0]+a2[1])
    c=0
else:
    a=(a1[1]+a2[0])
    c=1
acc_train=float(a)/float(len(pred_train))
print("训练数据的聚类准确率为：")
print(acc_train)
print(c)
b1 = [0, 0]
b2 = [0, 0]
for i in range(0, int(len(pred_test) / 2)):
    if pred_test[i] == 0:b1[0] = b1[0] + 1
    if pred_test[i] == 1:b1[1] = b1[1] + 1
for j in range(int(len(pred_test) / 2), int(len(pred_test))):
    if pred_test[j] == 0:b2[0] = b2[0] + 1
    if pred_test[j] == 1:b2[1] = b2[1] + 1
print(b1)
print(b2)
if((b1[0]+b2[1])>=(b1[1]+b2[0])):
    if (c==0):
        b=(b1[0]+b2[1])
    else:
        b = (b1[1] + b2[0])
else:
    if(c==1):
        b = (b1[1] + b2[0])
    else:
        b = (b1[0] + b2[1])
acc_test=float(b)/float(len(pred_test))
print("测试数据的聚类准确率为：")
print(acc_test)

#投票
def get_max(shuzu):
    s=[0,0]
    for i in range(0,100):
        if (shuzu[i]==0):s[0]=s[0]+1
        else:s[1]=s[1]+1
    if(s[0]>s[1]):return 0
    if(s[0]<s[1]):return 1
    if(s[0]==s[1]):return 2

pred_train_vot=np.arange(len(pred_train)/100)
print(len(pred_train_vot))
for b in range(0, len(pred_train_vot)):
    i=get_max(pred_train[b*100:(b+1)*100])
    if(i==2):pred_train_vot[b]=pred_train_vot[b-1]
    if (i == 0): pred_train_vot[b] = 0
    if (i == 1): pred_train_vot[b] = 1
print(pred_train_vot)
a1=[0,0]
a2=[0,0]
for i in range(0,int(len(pred_train_vot)/2)):
    if pred_train_vot[i]==0:a1[0]=a1[0]+1
    if pred_train_vot[i]==1:a1[1]=a1[1]+1
for j in range(int(len(pred_train_vot)/2),int(len(pred_train_vot))):
    if pred_train_vot[j] == 0: a2[0] = a2[0] + 1
    if pred_train_vot[j] == 1: a2[1] = a2[1] + 1
print(a1)
print(a2)
if((a1[0]+a2[1])>=(a1[1]+a2[0])):
    a=(a1[0]+a2[1])
    c=0
else:
    a=(a1[1]+a2[0])
    c=1
acc_train_vot=float(a)/float(len(pred_train_vot))
print("训练数据的投票后聚类准确率为：")
print(acc_train_vot)

pred_test_vot = np.arange(len(pred_test) / 100)
print(len(pred_test_vot))
for b in range(0, len(pred_test_vot)):
    i = get_max(pred_test[b * 100:(b + 1) * 100])
    if (i == 2): pred_test_vot[b] = pred_test_vot[b - 1]
    if (i == 0): pred_test_vot[b] = 0
    if (i == 1): pred_test_vot[b] = 1
print(pred_test_vot)
b1 = [0, 0]
b2 = [0, 0]
for i in range(0, int(len(pred_test_vot) / 2)):
    if pred_test_vot[i] == 0: b1[0] = b1[0] + 1
    if pred_test_vot[i] == 1: b1[1] = b1[1] + 1
for j in range(int(len(pred_test_vot) / 2), int(len(pred_test_vot))):
    if pred_test_vot[j] == 0: b2[0] = b2[0] + 1
    if pred_test_vot[j] == 1: b2[1] = b2[1] + 1
print(b1)
print(b2)
if((b1[0]+b2[1])>=(b1[1]+b2[0])):
    if (c==0):
        b=(b1[0]+b2[1])
    else:b = (b1[1] + b2[0])
else:
    if(c==1):
        b = (b1[1] + b2[0])
    else:b=(b1[0]+b2[1])
acc_test_vot = float(b) / float(len(pred_test_vot))
print("测试数据的投票后聚类准确率为：")
print(acc_test_vot)


t=[0,0]
for i in range(0,len(pred_tk)):
    if pred_tk[i]==0:t[0]=t[0]+1
    if pred_tk[i]==1:t[1]=t[1]+1
print(t)
if(c==0):acc_tk=float(t[0])/float(len(pred_tk))
if(c==1):acc_tk=float(t[1])/float(len(pred_tk))
print("other的准确率为：")
print(acc_tk)

pred_tk_vot = np.arange(len(pred_tk) / 100)
print(len(pred_tk_vot))
for b in range(0, len(pred_tk_vot)):
    i = get_max(pred_tk[b * 100:(b + 1) * 100])
    if (i == 2): pred_tk_vot[b] = pred_tk_vot[b - 1]
    if (i == 0): pred_tk_vot[b] = 0
    if (i == 1): pred_tk_vot[b] = 1
print(pred_tk_vot)
b1 = [0, 0]
for i in range(0, int(len(pred_tk_vot))):
    if pred_tk_vot[i] == 0: b1[0] = b1[0] + 1
    if pred_tk_vot[i] == 1: b1[1] = b1[1] + 1
if(c==0):acc_tk_vot=float(b1[0])/float(len(pred_tk_vot))
if(c==1):acc_tk_vot=float(b1[1])/float(len(pred_tk_vot))
print("投票后other的准确率为：")
print(acc_tk_vot)
k = 2
centroids, clusterAssment = KMeans1(train_mid, k)
showCluster(train_mid, k, centroids, clusterAssment)
