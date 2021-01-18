
# coding: utf-8

# # Variational Autoencoder
#
# Example is from the [Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html). According to the writer it's the simplest possible autoencoder.

# In[1]:

from sklearn.cluster import KMeans
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import pandas as pd
import os


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
    for j in ["0", "2Mhid"]:  # "1S", "2S"
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
    for j in ["0","1M"]:  # "1S", "2S"
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

# Load the dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
trainfile_array, testfile_array = file_array()
tk_files=file_array_other()
train_feature_all, train_label_all = read_data(trainfile_array)
test_feature_all, test_label_all = read_data(testfile_array)
tk_feature,tk_label=read_data(tk_files)

# image_size = x_train.shape[1]
# original_dim = image_size * image_size
#
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
x_train = train_feature_all.astype('float32')/ 73.0
x_test = test_feature_all.astype('float32') / 73.0
tk_feature=tk_feature.astype('float32')/73.0
print(x_train.shape)
print(x_test.shape)

batch_size = 128
original_dim = 270
# You would assume that 10 latent dimensions would be best for the MNIST dataset...
latent_dim = 2
epochs = 60
input_shape = (original_dim, )


# In[3]:


# Build encoder

# The input is mapped to a hidden layer "h" with dimension "intermediate_dim" and then mapped immediately to
# a latent layer with Gaussians (mean and log sigma variables). All subsequent layers are fully connected.
inputs = Input(input_shape)
h = Dense(128, activation='relu')(inputs)
z_mean = Dense(latent_dim)(inputs)
z_log_variance = Dense(latent_dim)(inputs)


# Given mean, $\mu$, and (log) sigma, $\log \sigma$, sample from a Normal distribution: $z \sim N(\mu, \sigma^2)$.
#
# Here Lambda is actually a layer. It accepts a vector with means and sigmas and returns a vector with samples
#
# Rather than directly sampling from $N(\mu,\sigma^2)$ we sample from $\epsilon \sim N(0,1)$ and calculate $z = \mu + \sigma \odot \epsilon$. Note that this function expects $\log \sigma$, not $\log \sigma^2$. The above representation is a product-wise multiplication by $\sigma$, but represents $N(\mu,\sigma^2)$.

# In[4]:


from keras.layers import Lambda
from keras import backend as K

def sampling(args):
    z_mean, z_log_variance = args
    z = np.hstack((z_mean, z_log_variance))
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    # It stated z_mean + K.exp(z_log_sigma) * epsilon in original code, but actually log variance was used
    # in Kullback-Leibler divergence. Hence I changed z_log_sigma -> z_log_variance and divide here by 2 in the
    # exponent, corresponds to sqrt(z_variance).
    return z

z = Lambda(sampling)([z_mean, z_log_variance])


# In[5]:


# instantiate encoder, from inputs to latent space
encoder = Model(inputs, [z_mean, z_log_variance, z])


# In[6]:


# Build decoder model
# The latent Gaussian variables are mapped to again a layer with dimension "intermediate_dim", finally the
# reconstruction is formed by mapping to original dimension.
# If properly trained, x_decoded_mean should be the same as x.
decoder_input = Input(shape=(latent_dim,))
decoder_h = Dense(128, activation='relu')(decoder_input)
decoder_output = Dense(original_dim, activation='sigmoid')(decoder_h)
#x_decoded_mean = decoder_mean(h1)

# Instantiate decoder
decoder = Model(decoder_input, decoder_output)


# In[8]:


# end-to-end autoencoder
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)


# The loss for the Variational Autoencoder is:
# * a "binary cross-entropy" loss between x and x' (the reconstructed x_decoded_mean)
# * a Kullback-Leibler divergence with the latent layer
#
# The Kullback-Leibler divergence between two multivariate normal distributions:
#
# $$D_{KL}(N_0,N_1) = 1/2 \left( \mathrm{tr }(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^T\Sigma_1^{-1} (\mu_1 - \mu_0) -k + \log \frac{\det \Sigma_1}{ \det \Sigma_0} \right)$$
#
# Here $\Sigma_0$ and $\Sigma_1$ are covariance matrices.
#
# In our case we compare a diagonal multivariate normal (one $\sigma$ scalar per variable) with a unit normal distribution. The trace of a matrix is just the sum over the diagonal. The determinant of a diagonal matrix is the product over the diagonal. The unit normal distribution: $\Sigma_1 = I$, $\mu_1=0$ (vector notation omitted).
#
# $$D_{KL}(N_0,N_1) = 1/2 \left( \sum_k ( \Sigma_0 ) + (-\mu_0)^T (-\mu_0) -k - \log  \prod_k (\Sigma_0) \right)$$
#
# And:
#
# $$D_{KL}(N_0,N_1) = 1/2 \left( \sum_k ( \Sigma_0 ) + \sum_k (\mu_i^2) + \sum_k ( -1 ) - \sum_k \log \Sigma_0 \right)$$
#
# Which leads to:
#
# $$D_{KL}(N_0,N_1) = -1/2 \sum_{i=1}^k 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2$$
#
# Here $N_0 = N(\mu_1,\ldots,\mu_k;\sigma_1,\ldots,\sigma_k)$ and $N_1 = N(0,I)$. We use the standard deviation $\sigma_i$ here rather than the (co)variance $\Sigma$.
#
# Note, it seems that z_log_sigma here actually represents $\log \sigma^2$. Is this correct?
#
# Rather than K.sum() for the KL divergence K_mean() is used. The entire loss is scaled with the original dimension. It would also have been an option to multiply the xend_loss with original_dim and then return the sum.

# In[9]:


from keras import objectives
from keras.losses import mse, binary_crossentropy

def vae_loss(x, x_reconstruction):
    xent_loss = binary_crossentropy(x, x_reconstruction) * original_dim
    # if we set kl_loss to 0 we get low values pretty immediate...
    # in 1 epoch, loss: 0.1557 - val_loss: 0.1401
    # in 50 epochs, loss: 0.0785 - val_loss: 0.0791
    kl_loss = - 0.5 * K.sum(1 + z_log_variance - K.square(z_mean) - K.exp(z_log_variance), axis=-1)
    return K.mean(xent_loss + kl_loss)

vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()


# In[11]:


steps_per_epoch=None
vae.fit(x_train, x_train,epochs=epochs,batch_size=batch_size,verbose = 1,validation_data=(x_test, x_test))
vae.save_weights('vae_mlp_CSI.h5')


# In[12]:


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs, _, _, = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[14]:


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(15, 18))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(15, 18))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

train_mid, _, _, = encoder.predict(x_train)
test_mid, _, _, = encoder.predict(x_test)
tk_mid, _, _, = encoder.predict(tk_feature)
print(train_mid)
print(test_mid)


kmeans=KMeans(n_clusters=2,n_init=20).fit(train_mid)
pred_train=kmeans.predict(train_mid)
print(pred_train)
pred_test=kmeans.predict(test_mid)
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