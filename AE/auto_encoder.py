#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.models import Model
import numpy as np
#pca和auto encoder原理网址
#https://blog.csdn.net/luoluonuoyasuolong/article/details/90711318


# In[4]:


#step 1 数据预处理
#这里需要说明一下，导入的原始数据shape为(60000,28,28),autoencoder使用(60000,28*28)，
#而且autoencoder属于无监督学习，所以只需要导入x_train和x_test.
(x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32')/255.0
# x_test = x_test.astype('float32')/255.0
#print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
#print(x_train.shape)


#step 2 向图片添加噪声
#添加噪声是为了让autoencoder更robust，不容易出现过拟合。
#add random noise
x_train_nosiy = x_train
x_test_nosiy = x_test
# x_train_nosiy = x_train + 0.3 * np.random.normal(loc=0., scale=1., size=x_train.shape)
# x_test_nosiy = x_test + 0.3 * np.random.normal(loc=0, scale=1, size=x_test.shape)
# x_train_nosiy = np.clip(x_train_nosiy, 0., 1.)
# x_test_nosiy = np.clip(x_test_nosiy, 0, 1.)
print(x_train_nosiy.shape, x_test_nosiy.shape)


#step 3 搭建网络结构
#分别构建encoded和decoded,然后将它们链接起来构成整个autoencoder。使用Model建模。
#build autoencoder model
input_img = Input(shape=(28*28,))
# encoded = Dense(100, activation='relu')(input_img)
# decoded = Dense(784, activation='sigmoid')(encoded)

encoded1 = Dense(500, activation='relu')(input_img)
encoded2 = Dense(128, activation='relu')(encoded1)
decoded1 = Dense(500, activation='relu')(encoded1)
decoded2 = Dense(784, activation='relu')(decoded1)

autoencoder = Model(input=input_img, output=decoded2)


#step 4 compile
#因为这里是让解压后的图片和原图片做比较， loss使用的是binary_crossentropy。

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

#step 5 train
#指定epochs，batch_size，可以使用validation_data,keras训练的时候不会使用它，而是用来做模型评价
autoencoder.fit(x_train_nosiy, x_train, epochs=2, batch_size=128, verbose=1, validation_data=(x_test, x_test))

#step 6 对比一下解压缩后的图片和原图片
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#decoded test images
decoded_img = autoencoder.predict(x_test_nosiy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #noisy data
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test_nosiy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #predict
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    #original
    ax = plt.subplot(3, n, i+1+2*n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
plt.show()


# In[ ]:




