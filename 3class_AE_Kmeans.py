#一个人的训练模型去测试这个人的没有训练过的数据


from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
from tensorflow.python.keras.models import load_model
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    testfile = []
    for j in ["0","2Mhid", "3M"]:  # "1S", "2S"
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
lin=120
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
                                         int(csvdata.shape[0] / 2) +lin, 2)])#取中心点处左右分布数据
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

trainfile_array, testfile_array = file_array()#
tk_files=file_array_other()
train_feature, train_label = read_data(trainfile_array)
test_feature, test_label = read_data(testfile_array)
tk_feature,tk_label=read_data(tk_files)


train_feature = train_feature.astype('float32')/73.0
test_feature = test_feature.astype('float32')/73.0
tk_feature=tk_feature.astype('float32')/73.0
# train_feature = (train_feature.astype('float32')-37)/36.0
# test_feature = (test_feature.astype('float32')-37)/36.0
# tk_feature=(tk_feature.astype('float32')-37)/36.0
# train_feature=pow(train_feature, 2.0/3)
# test_feature = pow(test_feature, 2.0/3)
#train_feature_nosiy=train_feature
#test_feature_nosiy=test_feature
# train_feature_nosiy=pow(train_feature, 2.0/3)
# test_feature_nosiy = pow(test_feature, 2.0/3)

# train_feature_nosiy = train_feature
# test_feature_nosiy = test_feature
train_feature_nosiy = train_feature+0.005 * np.random.normal(loc=0., scale=1., size=train_feature.shape)
test_feature_nosiy = test_feature+0.005 * np.random.normal(loc=0., scale=1., size=test_feature.shape)
# train_feature_nosiy = np.clip(train_feature_nosiy, 0., 1.)
# test_feature_nosiy = np.clip(test_feature_nosiy, 0, 1.)
input = Input(shape=(270,))

encoded1 = Dense(128, activation='relu')(input)
#encoded1 = Dense(128, activation='relu')(encoded1)
encoded2 = Dense(2)(input)
decoded1 = Dense(128, activation='relu')(encoded2)
#decoded1 = Dense(128, activation='relu')(decoded1)
#decoded1 = Dense(128, activation='relu')(decoded1)
#decoded2 = Dense(270, activation='sigmoid')(decoded1)
decoded2 = Dense(270, activation='sigmoid')(decoded1)
#decoded2 = Dense(270, activation='relu')(decoded1)

autoencoder = Model(input=input, output=decoded2)
#print(autoencoder.inputs)
autoencoder_mid = Model(inputs=input, outputs=encoded2)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
autoencoder.fit(train_feature_nosiy, train_feature, epochs=60, batch_size=128, verbose=1, validation_data=(test_feature_nosiy, test_feature))

# autoencoder.save("model")
# model = load_model("model")





#decoded test images
train_mid = autoencoder_mid.predict(train_feature_nosiy)
test_mid = autoencoder_mid.predict(test_feature_nosiy)
tk_mid = autoencoder_mid.predict(tk_feature)
print(train_mid)
print(test_mid)
decoded_img = autoencoder.predict(test_feature_nosiy)
#decoded_img1 = autoencoder.encoder(x_test_nosiy)
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     #noisy data
#     ax = plt.subplot(3, n, i+1)
#     plt.imshow(test_feature_nosiy[i].reshape(15, 18))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     #predict
#     ax = plt.subplot(3, n, i+1+n)
#     plt.imshow(decoded_img[i].reshape(15, 18))
#     plt.gray()
#     ax.get_yaxis().set_visible(False)
#     ax.get_xaxis().set_visible(False)
#     #original
#     ax = plt.subplot(3, n, i+1+2*n)
#     plt.imshow(test_feature[i].reshape(15, 18))
#     plt.gray()
#     ax.get_yaxis().set_visible(False)
#     ax.get_xaxis().set_visible(False)
# plt.show()

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



kmeans=KMeans(n_clusters=3,n_init=50).fit(train_mid)
pred_train=kmeans.predict(train_mid)
print(pred_train)
pred_test=kmeans.predict(test_mid)
print(pred_test)
pred_tk=kmeans.predict(tk_mid)

a1=[0,0,0]
a2=[0,0,0]
a3=[0,0,0]
for i in range(0,int(len(pred_train)/3)):
    if pred_train[i]==0:a1[0]=a1[0]+1
    if pred_train[i]==1:a1[1]=a1[1]+1
    if pred_train[i]==2:a1[2]=a1[2]+1
for j in range(int(len(pred_train)/3),int(len(pred_train)*2/3)):
    if pred_train[j] == 0: a2[0] = a2[0] + 1
    if pred_train[j] == 1: a2[1] = a2[1] + 1
    if pred_train[j] == 2: a2[2] = a2[2] + 1
for k in range(int(len(pred_train)*2/3),int(len(pred_train))):
    if pred_train[k] == 0: a3[0] = a3[0] + 1
    if pred_train[k] == 1: a3[1] = a3[1] + 1
    if pred_train[k] == 2: a3[2] = a3[2] + 1
print(a1)
print(a2)
print(a3)

b1 = [0, 0,0]
b2 = [0, 0,0]
b3=[0,0,0]
for i in range(0, int(len(pred_test) / 3)):
    if pred_test[i] == 0:b1[0] = b1[0] + 1
    if pred_test[i] == 1:b1[1] = b1[1] + 1
    if pred_test[i] == 2: b1[2] = b1[2] + 1
for j in range(int(len(pred_test) / 3), int(len(pred_test)*2/3)):
    if pred_test[j] == 0:b2[0] = b2[0] + 1
    if pred_test[j] == 1:b2[1] = b2[1] + 1
    if pred_test[j] == 2: b2[2] = b2[2] + 1
for j in range(int(len(pred_test) *2/ 3), int(len(pred_test))):
    if pred_test[j] == 0:b3[0] = b3[0] + 1
    if pred_test[j] == 1:b3[1] = b3[1] + 1
    if pred_test[j] == 2: b3[2] = b3[2] + 1
print(b1)
print(b2)
print(b3)


#投票
def get_max(shuzu):
    s=[0,0,0]
    for i in range(0,lin):
        if (shuzu[i]==0):s[0]=s[0]+1
        if (shuzu[i] == 1):s[1] = s[1] + 1
        if (shuzu[i] == 2):s[2] = s[2] + 1
    if(s[0]>=s[1]):
        if(s[0]>=s[2]):
            return 0
    if(s[1]>=s[0]):
        if (s[1] >= s[2]):
            return 1
    if(s[2]>=s[0]):
        if(s[2]>=s[1]):
            return 2

pred_train_vot=np.arange(len(pred_train)/lin)
print(len(pred_train_vot))
for b in range(0, len(pred_train_vot)):
    i=get_max(pred_train[b*lin:(b+1)*lin])
    if (i == 0): pred_train_vot[b] = 0
    if (i == 1): pred_train_vot[b] = 1
    if (i == 2): pred_train_vot[b] = 2
print(pred_train_vot)


a1=[0,0,0]
a2=[0,0,0]
a3=[0,0,0]
for i in range(0,int(len(pred_train_vot)/3)):
    if pred_train_vot[i]==0:a1[0]=a1[0]+1
    if pred_train_vot[i]==1:a1[1]=a1[1]+1
    if pred_train_vot[i]==2:a1[2]=a1[2]+1
for j in range(int(len(pred_train_vot)/3),int(len(pred_train_vot)*2/3)):
    if pred_train_vot[j] == 0: a2[0] = a2[0] + 1
    if pred_train_vot[j] == 1: a2[1] = a2[1] + 1
    if pred_train_vot[j] == 2: a2[2] = a2[2] + 1
for k in range(int(len(pred_train_vot)*2/3),int(len(pred_train_vot))):
    if pred_train_vot[k] == 0: a3[0] = a3[0] + 1
    if pred_train_vot[k] == 1: a3[1] = a3[1] + 1
    if pred_train_vot[k] == 2: a3[2] = a3[2] + 1
print(a1)
print(a2)
print(a3)


pred_test_vot = np.arange(len(pred_test) / lin)
print(len(pred_test_vot))
for b in range(0, len(pred_test_vot)):
    i = get_max(pred_test[b * lin:(b + 1) * lin])
    if (i == 0): pred_test_vot[b] = 0
    if (i == 1): pred_test_vot[b] = 1
    if (i == 2): pred_test_vot[b] = 2
print(pred_test_vot)


b1 = [0, 0,0]
b2 = [0, 0,0]
b3=[0,0,0]
for i in range(0, int(len(pred_test_vot) / 3)):
    if pred_test_vot[i] == 0:b1[0] = b1[0] + 1
    if pred_test_vot[i] == 1:b1[1] = b1[1] + 1
    if pred_test_vot[i] == 2: b1[2] = b1[2] + 1
for j in range(int(len(pred_test_vot) / 3), int(len(pred_test_vot)*2/3)):
    if pred_test_vot[j] == 0:b2[0] = b2[0] + 1
    if pred_test_vot[j] == 1:b2[1] = b2[1] + 1
    if pred_test_vot[j] == 2: b2[2] = b2[2] + 1
for j in range(int(len(pred_test_vot) *2/ 3), int(len(pred_test_vot))):
    if pred_test_vot[j] == 0:b3[0] = b3[0] + 1
    if pred_test_vot[j] == 1:b3[1] = b3[1] + 1
    if pred_test_vot[j] == 2: b3[2] = b3[2] + 1
print(b1)
print(b2)
print(b3)



k = 2
centroids, clusterAssment = KMeans1(train_mid, k)
showCluster(train_mid, k, centroids, clusterAssment)
