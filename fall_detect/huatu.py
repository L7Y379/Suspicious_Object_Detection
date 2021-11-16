#画图
#列归一
import os
import pandas as pd
import numpy as np
from keras.layers import LSTM,Input, Dense,Flatten,MaxPooling2D,TimeDistributed,Bidirectional, Conv2D
from keras.models import Sequential, Model
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

def train_t(train_feature,test_feature,train_label,model_path):
    print("train_feature" + str(train_feature.shape))
    print("test_feature" + str(test_feature.shape))
    print("train_label" + str(train_label.shape))

    #全局归化为0~1
    # a=train_feature.reshape(int(train_feature.shape[0] / 200),200*270)
    # a=a.T
    # min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    # train_feature = min_max_scaler.fit_transform(a)
    # print(train_feature.shape)
    # train_feature=train_feature.T
    #
    # a = test_feature.reshape(int(test_feature.shape[0] / 200), 200 * 270)
    # a = a.T
    # min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    # test_feature = min_max_scaler.fit_transform(a)
    # print(test_feature.shape)
    # test_feature = test_feature.T
    # train_feature = train_feature.reshape([train_feature.shape[0], 200, img_rows, img_cols])
    # train_feature = np.expand_dims(train_feature, axis=4)
    # test_feature = test_feature.reshape([test_feature.shape[0], 200, img_rows, img_cols])
    # test_feature = np.expand_dims(test_feature, axis=4)


    # 列归化为0~1
    # min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    # train_feature = min_max_scaler.fit_transform(train_feature)
    # test_feature=min_max_scaler.fit_transform(test_feature)

    # 列归化为0~1（针对每个样本单独归一）
    min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    for i in range(int(train_feature.shape[0]/200)):
        train_feature[i*200:(i+1)*200]=min_max_scaler.fit_transform(train_feature[i*200:(i+1)*200])
    for i in range(int(test_feature.shape[0]/200)):
        test_feature[i*200:(i+1)*200]=min_max_scaler.fit_transform(test_feature[i*200:(i+1)*200])
    train_feature = train_feature.reshape([int(train_feature.shape[0] / 200), 200, 270])
    train_feature = np.expand_dims(train_feature, axis=4)
    test_feature = test_feature.reshape([int(test_feature.shape[0] / 200), 200, 270])
    test_feature = np.expand_dims(test_feature, axis=4)
    plt.plot(train_feature[0])
    plt.show()
    plt.plot(train_feature[5])
    plt.show()
    plt.plot(test_feature[0])
    plt.show()
    plt.plot(test_feature[5])
    plt.show()

def getlabel(path):
    path2=path.replace(".csv",".dat")
    col_name = ['path','label']
    filepath = "/content/drive/MyDrive/data/fall/labels/labels.csv"
    csv_data = pd.read_csv(filepath, names=col_name, header=None)
    label = -1
    for index, row in csv_data.iterrows():
        # print(row)
        if row["path"] == path or row["path"] == path2:
            # print(111)
            label = row["label"]
    # print("labels:",labels)
    print("label", label)
    return label
#直接取预处理后的数据训练训练
def model_train_test(dirname,dirname2, model_path):
    k = 0 #标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    for i in range(0,len(dataList)):
        path = os.path.join(dirname,dataList[i])
        if os.path.isfile(path):
            temp_data=pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            if k == 0:
                raw_data=temp_data
                label = getlabel(dataList[i])
                k = 1
            else:
                raw_data = np.row_stack((raw_data, temp_data))
                label = np.row_stack((label, getlabel(dataList[i])))
    label=np_utils.to_categorical(label)
    print("raw_data:",raw_data.shape)
    print("label:", label.shape)
    data=raw_data
    label = label.astype(np.float32)

    k = 0  # 标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname2)
    for i in range(0, len(dataList)):
        path = os.path.join(dirname2, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            if k == 0:
                raw_data = temp_data
                label2 = getlabel(dataList[i])
                k = 1
            else:
                raw_data = np.row_stack((raw_data, temp_data))
                label2 = np.row_stack((label2, getlabel(dataList[i])))
    label2 = np_utils.to_categorical(label2)
    print("raw_data:", raw_data.shape)
    print("label2:", label2.shape)
    data2 = raw_data
    label2 = label2.astype(np.float32)
    train_t(data,data2,label, model_path)
def train_test():
    print("train_test已调用")

    model_path = "/content/drive/MyDrive/data/fall/1112_lie1.h5"
    dirPath="/content/drive/MyDrive/data/fall/1112_pre"
    dirPath2 = "/content/drive/MyDrive/data/fall/1115_pre"
    model_train_test(dirPath,dirPath2, model_path)

#test_walk()
train_test()