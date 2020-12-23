import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 加载数据
def file_array():
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
    print(testfile);
    return trainfile, testfile

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
        idx = np.array([j for j in range(int(csvdata.shape[0] / 2)-5 ,
                                         int(csvdata.shape[0] / 2) +5, 1)])#取中心点处左右分布数据
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
    np.random.shuffle(feature)
    return np.array(feature[:, :270]), np.array(feature[:, 270:])
    #return np.array(feature[:, 134:136]), np.array(feature[:, 134:136])

trainfile_array, testfile_array = file_array()#
train_feature, train_label = read_data(trainfile_array)
pca = PCA(n_components=2)
train_feature_PCA=pca.fit_transform(train_feature)#PCA降维为2维
print(train_feature_PCA.shape, train_label.shape)
test_feature, test_label = read_data(testfile_array)
test_feature_PCA=pca.fit_transform(test_feature)#PCA降维为2维
print(test_feature_PCA.shape, test_label.shape)
#print(csvdata)


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



#print(train_feature_PCA)
k = 2
centroids, clusterAssment = KMeans1(train_feature_PCA, k)
kmeans=KMeans(n_clusters=2).fit(train_feature_PCA)
pred=kmeans.predict(train_feature_PCA)

print(pred)
#print(pred[200:])

pred2=kmeans.predict(test_feature_PCA)

print(pred2)
#print(pred2[100:])
showCluster(train_feature_PCA, k, centroids, clusterAssment)