import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    testfile = []
    for j in ["0", "1M","2M"]:  # "1S", "2S"
        for i in [i for i in range(0, 30)]:
            fn = filepath + "ljy-2.5-M/" + "ljy-" + str(j) + "-" + str(i) + filetype
            filenames += [fn]
        #np.random.shuffle(filenames)
        if (j == "0"):
            trainfile += filenames[:30]
            testfile += filenames[20:]
        if (j == "1M"):
            trainfile += filenames[:0]
            testfile += filenames[22:]
        if (j == "2M"):
            trainfile += filenames[:0]
            testfile += filenames[23:]
        filenames = []
    trainfile = np.array(trainfile)#20*2
    testfile = np.array(testfile)#5*2
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
        #np.random.shuffle(filenames)
    filenames = np.array(filenames)#20*2
    return filenames
lin=120
ww=1
lin2=int((lin*2)/ww)
print(lin2)
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
            #label = np.concatenate((label, temp_label), axis=0)
    #data = np.concatenate((feature, label), axis=1)
    #np.random.shuffle(feature)
    return np.array(feature[:, :270]), np.array(feature[:, 270:])
    #return np.array(feature[:, 134:136]), np.array(feature[:, 134:136])

trainfile_array, testfile_array = file_array()#
print(trainfile_array)
print(testfile_array)

train_feature, train_label = read_data(trainfile_array)
test_feature, test_label = read_data(testfile_array)
print(train_feature.shape)
print(len(train_feature))

kmeans=KMeans(n_clusters=1,n_init=50)
pred_train=kmeans.fit_predict(train_feature)
print(kmeans.cluster_centers_.shape)
print(kmeans.cluster_centers_)
for i in range(0,int(len(train_feature)/240)):
    k=np.mean(train_feature[i*240:(i+1)*240])-np.mean(train_feature)
    print(k)
print(np.mean(train_feature))
print(np.median(train_feature))
print(np.min(test_feature))
print(np.max(test_feature))
a=np.min(np.concatenate((train_feature,test_feature), axis=0))
b=np.max(np.concatenate((train_feature,test_feature), axis=0))
print(a)
print(b)

train_feature=train_feature-kmeans.cluster_centers_
print(train_feature)
train_feature=np.square(train_feature)
print(train_feature)
print(train_feature.shape)
train_feature=np.sum(train_feature,axis=1)
print(train_feature)
print(train_feature.shape)
train_feature=np.sqrt(train_feature)
print(train_feature)
print(train_feature.shape)
k = np.arange(len(train_feature) / 240)
print(k)
for i in range(0,int(len(train_feature)/240)):
    k[i] = np.mean(train_feature[i * 240:(i + 1) * 240])
    print(k[i])
# train_feature = (train_feature.astype('float32')-np.min(np.concatenate((train_feature,test_feature), axis=0)))/(np.max(np.concatenate((train_feature,test_feature), axis=0))-np.min(np.concatenate((train_feature,test_feature), axis=0)))
# test_feature = (test_feature.astype('float32')-np.min(np.concatenate((train_feature,test_feature), axis=0)))/(np.max(np.concatenate((train_feature,test_feature), axis=0))-np.min(np.concatenate((train_feature,test_feature), axis=0)))


