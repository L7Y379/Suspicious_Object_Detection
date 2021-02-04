from keras.models import Model
import numpy as np
from tensorflow.python.keras.models import load_model
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def file_array():
    filepath = 'D:/my bad/Suspicious object detection/data/CSV/'
    filetype = '.csv'
    filenames = []
    trainfile = []
    trainfile2 = []
    testfile = []
    testfile2 = []
    for name in ['zb', 'ljy', 'tk', 'czn']:
        for j in ["0"]:  # "1S", "2S"
            for i in [i for i in range(0, 30)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]

    trainfile += filenames[:120]
    trainfile = np.array(trainfile)
    feature, lable = read_data(trainfile)
    k1 = np.arange(30)
    for i in range(0, 30):
        k1[i] = np.square(np.mean(feature[i * 240:(i + 1) * 240]) - np.mean(feature[:7200]))
    trainfilek1 = trainfile[np.argsort(k1)]
    trainfilek1 = trainfilek1[:25]
    np.random.shuffle(trainfilek1)
    trainfilek1 = np.array(trainfilek1)

    trainfilek2 = trainfile[30:60]
    k2 = np.arange(30)
    for i in range(0, 30):
        k2[i] = np.square(np.mean(feature[(i + 30) * 240:(i + 1 + 30) * 240]) - np.mean(feature[7200:14400]))
    trainfilek2 = trainfilek2[np.argsort(k2)]
    trainfilek2 = trainfilek2[:25]
    np.random.shuffle(trainfilek2)
    trainfilek2 = np.array(trainfilek2)

    trainfilek3 = trainfile[60:90]
    k3 = np.arange(30)
    for i in range(0, 30):
        k3[i] = np.square(np.mean(feature[(i + 60) * 240:(i + 1 + 60) * 240]) - np.mean(feature[14400:21600]))
    trainfilek3 = trainfilek3[np.argsort(k3)]
    trainfilek3 = trainfilek3[:25]
    np.random.shuffle(trainfilek3)
    trainfilek3 = np.array(trainfilek3)

    trainfilek4 = trainfile[90:120]
    k4 = np.arange(30)
    for i in range(0, 30):
        k4[i] = np.square(np.mean(feature[(i + 90) * 240:(i + 1 + 90) * 240]) - np.mean(feature[21600:28800]))
    trainfilek4 = trainfilek4[np.argsort(k3)]
    trainfilek4 = trainfilek4[:25]
    np.random.shuffle(trainfilek4)
    trainfilek4 = np.array(trainfilek4)

    trainfile = np.concatenate((trainfilek1, trainfilek2, trainfilek3, trainfilek4), axis=0)

    testfile = np.concatenate((trainfile[20:25], trainfile[45:50], trainfile[70:75], trainfile[95:100]), axis=0)
    trainfile = np.concatenate((trainfile[:20], trainfile[25:45], trainfile[50:70], trainfile[75:95]), axis=0)
    for name in ['zb', 'ljy', 'tk', 'czn']:
        for j in ["0", "1M", "2M"]:  # "1S", "2S"
            for i in [i for i in range(0, 25)]:
                fn = filepath + name + "-2.5-M/" + name + "-" + str(j) + "-" + str(i) + filetype
                filenames += [fn]
            np.random.shuffle(filenames)
            if (j == "1M"):
                trainfile2 += filenames[:10]
                testfile2 += filenames[22:]
            if (j == "2M"):
                trainfile2 += filenames[:10]
                testfile2 += filenames[23:]
            filenames = []
    trainfile2 = np.array(trainfile2)  # 20*2
    testfile2 = np.array(testfile2)  # 20*2
    trainfile = np.concatenate((trainfile, trainfile2), axis=0)
    testfile = np.concatenate((testfile, testfile2), axis=0)
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
lin=120
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
            #label = np.concatenate((label, temp_label), axis=0)
    #data = np.concatenate((feature, label), axis=1)
    #np.random.shuffle(feature)
    return np.array(feature[:, :270]), np.array(feature[:, 270:])
    #return np.array(feature[:, 134:136]), np.array(feature[:, 134:136])

trainfile_array, testfile_array = file_array()#
print(trainfile_array)
print(testfile_array)
print(trainfile_array.shape)
print(testfile_array.shape)
train_feature, train_label = read_data(trainfile_array)
test_feature, test_label = read_data(testfile_array)