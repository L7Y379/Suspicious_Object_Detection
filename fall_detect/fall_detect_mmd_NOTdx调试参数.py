#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import math
import os
from torch.autograd import Variable
from torch import nn
from keras.utils import np_utils
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sampleNum=300
batch_size = 50
lr = 0.001
class_num = 2
feature_len = 270
epochs = 100
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
     source: 源域数据，行表示样本数目，列表示样本数据维度
     target: 目标域数据 同source
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
  sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    # 求矩阵的行数，即两个域的的样本总数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance_square = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance_square) / (n_samples**2-n_samples)
    # 多核MMD
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # print(bandwidth_list)
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)
def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
     source: 源域数据，行表示样本数目，列表示样本数据维度
     target: 目标域数据 同source
     kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
     kernel_num: 取不同高斯核的数量
     fix_sigma: 是否固定，如果固定，则为单核MMD
 Return:
  loss: MMD loss
    '''
    source_num = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    target_num = int(target.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = torch.mean(kernels[:source_num, :source_num])
    YY = torch.mean(kernels[source_num:, source_num:])
    XY = torch.mean(kernels[:source_num, source_num:])
    YX = torch.mean(kernels[source_num:, :source_num])
    loss = XX + YY -XY - YX
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算
def cal_mmd(source, target, slabel, tlabel):
    loss = torch.Tensor([0]).cuda()
    a = 0.5
    b = 0.02
    for i in range(6):
        s = source[slabel==i]
        t = target[tlabel==i]
        nt = target[tlabel!=i]
        if len(s) != 0 and len(t) != 0:
            loss += (mmd_rbf(s, t)*a)
        if len(s) != 0 and len(nt) != 0:
            loss -= (mmd_rbf(s, nt)*b)
    return loss
class Feature_extractor(nn.Module):
    def __init__(self, feature_len):
        super(Feature_extractor, self).__init__()
        self.feature_extractor = nn.Sequential(
          nn.Conv1d(in_channels = feature_len, out_channels=128, kernel_size=20),#feature_len270
          #nn.BatchNorm1d(200),
          nn.ReLU(),

          nn.Conv1d(in_channels = 128, out_channels=64, kernel_size=20),
          #nn.BatchNorm1d(150),
          nn.ReLU(),

          nn.Conv1d(in_channels = 64, out_channels=32, kernel_size=20),
          #nn.BatchNorm1d(100),
          nn.ReLU(),

          nn.Conv1d(in_channels = 32, out_channels=10, kernel_size=20),
          #nn.BatchNorm1d(50),
          nn.Flatten()
        )

    def forward(self, ipt):
        feature = self.feature_extractor(ipt)

        return feature
class ClassClassfier(nn.Module):
    def __init__(self, class_num):
        super(ClassClassfier, self).__init__()
        self.class_classifier = nn. Sequential(
          nn.Linear(2050, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Linear(64, class_num),
          nn.Softmax(dim=1)
        )

    def forward(self, ipt):
        class_out = self.class_classifier(ipt)

        return class_out
class F_C():
    def __init__(self,feature_extractor,classClassfier):
        super(F_C, self).__init__()
        self.feature_extractor = torch.nn.Sequential()
        self.feature_extractor.add_module("feature_extractor",feature_extractor)
        self.classClassfier = torch.nn.Sequential()
        self.classClassfier.add_module("classClassfier", classClassfier)
    def pre(self, ipt):
        feature = self.feature_extractor(ipt)
        class_out = self.classClassfier(feature)
        return class_out
def getlabel(path):
    path2=path.replace(".csv",".dat")
    col_name = ['path','label']
    filepath = "D:/my bad/Suspicious object detection/Suspicious_Object_Detection/yue/fall_detect/labels.csv"
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
def model_train_test(dirname,dirname2, model_path):
    k = 0 #标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    dataList.sort()
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
    dataList.sort()
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
    return data,data2,label, model_path
def model_train_test2(dirname,dirname2,dirPath_test, model_path):
    k = 0 #标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    dataList.sort()
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
    #label=np_utils.to_categorical(label)
    print("raw_data:",raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    #label = label.astype(np.float32)

    dataList = os.listdir(dirname2)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirname2, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            raw_data = np.row_stack((raw_data, temp_data))
            label = np.row_stack((label, getlabel(dataList[i])))
    label = np_utils.to_categorical(label)
    print("raw_data:", raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    data = raw_data
    label = label.astype(np.float32)


    k = 0  # 标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirPath_test)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirPath_test, dataList[i])
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
    print("label2:", label2)
    data2 = raw_data
    label2 = label2.astype(np.float32)
    return data,data2,label, model_path
def model_train_test3(dirname,dirname2,dirname3,dirPath_test, model_path):
    k = 0 #标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    dataList.sort()
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
    #label=np_utils.to_categorical(label)
    print("raw_data:",raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    #label = label.astype(np.float32)

    dataList = os.listdir(dirname2)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirname2, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            raw_data = np.row_stack((raw_data, temp_data))
            label = np.row_stack((label, getlabel(dataList[i])))
    #label = np_utils.to_categorical(label)
    print("raw_data:", raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    #data = raw_data
    #label = label.astype(np.float32)

    dataList = os.listdir(dirname3)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirname3, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            raw_data = np.row_stack((raw_data, temp_data))
            label = np.row_stack((label, getlabel(dataList[i])))
    label = np_utils.to_categorical(label)
    print("raw_data:", raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    data = raw_data
    label = label.astype(np.float32)


    k = 0  # 标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirPath_test)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirPath_test, dataList[i])
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
    print("label2:", label2)
    data2 = raw_data
    label2 = label2.astype(np.float32)
    return data,data2,label, model_path
def model_train_test4(dirname,dirname2,dirname3,dirname4,dirPath_test, model_path):
    k = 0 #标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    dataList.sort()
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
    #label=np_utils.to_categorical(label)
    print("raw_data:",raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    #label = label.astype(np.float32)

    dataList = os.listdir(dirname2)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirname2, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            raw_data = np.row_stack((raw_data, temp_data))
            label = np.row_stack((label, getlabel(dataList[i])))
    #label = np_utils.to_categorical(label)
    print("raw_data:", raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    #data = raw_data
    #label = label.astype(np.float32)

    dataList = os.listdir(dirname3)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirname3, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            raw_data = np.row_stack((raw_data, temp_data))
            label = np.row_stack((label, getlabel(dataList[i])))
    # label = np_utils.to_categorical(label)
    print("raw_data:", raw_data.shape)
    print("label:", label.shape)
    print("label:", label)

    dataList = os.listdir(dirname4)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirname4, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            raw_data = np.row_stack((raw_data, temp_data))
            label = np.row_stack((label, getlabel(dataList[i])))
    label = np_utils.to_categorical(label)
    print("raw_data:", raw_data.shape)
    print("label:", label.shape)
    print("label:", label)
    data = raw_data
    label = label.astype(np.float32)


    k = 0  # 标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirPath_test)
    dataList.sort()
    for i in range(0, len(dataList)):
        path = os.path.join(dirPath_test, dataList[i])
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
    print("label2:", label2)
    data2 = raw_data
    label2 = label2.astype(np.float32)
    return data,data2,label, model_path
def train_test():
    print("train_test已调用")
    model_path = "D:/my bad/Suspicious object detection/Suspicious_Object_Detection/yue/fall_detect/models/feacturer.pkl"
    dirPath = "D:/my bad/Suspicious object detection/data/fall/notdx_newwalk/1117_pre"
    dirPath2 = "D:/my bad/Suspicious object detection/data/fall/notdx_newwalk/1119_pre"
    dirPath3 = "D:/my bad/Suspicious object detection/data/fall/notdx_newwalk/1123_pre"
    dirPath4 = "D:/my bad/Suspicious object detection/data/fall/notdx_newwalk/1203_pre"
    dirPath_test = "D:/my bad/Suspicious object detection/data/fall/notdx_newwalk/1202_pre"
    train_feature, test_feature, train_label, model_path=model_train_test(dirPath,dirPath_test, model_path)
    #train_feature, test_feature, train_label, model_path=model_train_test2(dirPath,dirPath2, dirPath_test,model_path)
    #train_feature, test_feature, train_label, model_path=model_train_test3(dirPath, dirPath2,dirPath3, dirPath_test, model_path)
    #train_feature, test_feature, train_label, model_path=model_train_test4(dirPath, dirPath2, dirPath3,dirPath4, dirPath_test, model_path)
    print("train_feature" + str(train_feature.shape))
    print("test_feature" + str(test_feature.shape))
    print("train_label" + str(train_label.shape))

    # 全局归化为0~1
    a = train_feature.reshape(int(train_feature.shape[0] / sampleNum), sampleNum * 270)
    a = a.T
    min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    train_feature = min_max_scaler.fit_transform(a)
    print(train_feature.shape)
    train_feature = train_feature.T

    a = test_feature.reshape(int(test_feature.shape[0] / sampleNum), sampleNum * 270)
    a = a.T
    min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    test_feature = min_max_scaler.fit_transform(a)
    print(test_feature.shape)
    test_feature = test_feature.T

    train_feature = train_feature.reshape([train_feature.shape[0], sampleNum, 270])
    train_feature = np.swapaxes(train_feature, 1, 2)
    test_feature = test_feature.reshape([test_feature.shape[0], sampleNum, 270])
    test_feature = np.swapaxes(test_feature, 1, 2)

    return train_feature, test_feature, train_label, model_path
def train():
    train_feature, test_feature, train_label, model_path=train_test()
    print("train_feature" + str(train_feature.shape))
    print("test_feature" + str(test_feature.shape))
    print("train_label" + str(train_label.shape))
    feature_extractor = Feature_extractor(feature_len).to(device)
    classClassfier = ClassClassfier(class_num).to(device)
    loss_class = nn.CrossEntropyLoss()

    optimizer_feature_extractor = optim.Adam(feature_extractor.parameters(), lr)
    optimizer_classClassfier = optim.Adam(classClassfier.parameters(), lr)
    loss_min=10000
    for epoch in range(1, 1 + epochs):
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        mmd_loss = torch.tensor(0.0)
        mmd_loss = Variable(mmd_loss, requires_grad=True)
        idx = np.random.randint(0, train_feature.shape[0], batch_size)
        imgs = train_feature[idx]
        source_item_csi_data = torch.tensor(imgs, dtype=torch.float32).to(device)
        #source_item_csi_data=torch.from_numpy(imgs).double().to(device)
        source_item_label=torch.from_numpy(train_label[idx]).to(device)
        source_item_label = torch.argmax(source_item_label, 1)
        optimizer_feature_extractor.zero_grad()
        optimizer_classClassfier.zero_grad()
        feature = feature_extractor(source_item_csi_data)
        class_out = classClassfier(feature)
        mmd = 0
        if(mmd==1):
            idx2 = np.random.randint(0, test_feature.shape[0], batch_size)
            t_img = torch.tensor(test_feature[idx2], dtype=torch.float32).to(device)
            feature_target = feature_extractor(t_img)
            mmd_loss = mmd_rbf(feature, feature_target)
            loss = loss_class(class_out, source_item_label) + mmd_loss * lambd
        if (mmd==0):
            loss = loss_class(class_out, source_item_label)
        loss.backward()
        optimizer_feature_extractor.step()
        optimizer_classClassfier.step()

        if epoch % 10 == 0:
            print(loss)
            n = 0
            a_all = np.zeros((3, 2))  # 3个动作：第一个跌倒，后两个个非跌倒
            for o in range(5):  # 四个人的数据
                # print(test_feature.shape)
                non_mid =feature_extractor(torch.tensor(test_feature[o * 20:(o + 1) * 20], dtype=torch.float32).to(device))# 每个人20条数据，10条跌倒，10条非跌倒
                non_pre = classClassfier(non_mid)  # (20,2)
                m = 0
                a = np.zeros((3, 2))
                for i in range(4):
                    for k in range(5):
                        x = torch.argmax(non_pre[i * 5 + k])
                        if (i == 0 or i == 1):
                            a[0][x] = a[0][x] + 1
                            a_all[0][x] = a_all[0][x] + 1
                        else:
                            a[i - 1][x] = a[i - 1][x] + 1
                            a_all[i - 1][x] = a_all[i - 1][x] + 1
                        if ((x == 0 and i <= 1) or (x == 1 and i >= 2)):
                            m = m + 1
                            n = n + 1
                acc = float(m) / float(len(non_pre))
                print("源" + str(o + 1) + "测试数据准确率：" + str(acc))
                print(a)
            ac = float(n) / float(100)
            k1 = ac
            print("源平均测试数据准确率：" + str(ac))
            print("精度：" + str(a_all[0:1, 0:1] / (a_all[0:1, 0:1] + a_all[1:2, 0:1] + a_all[2:3, 0:1])))
            print("召回率：" + str(a_all[0:1, 0:1] / (a_all[0:1, 0:1] + a_all[0:1, 1:2])))
            print(a_all)

            if (loss <= loss_min):
                loss = loss_min
                torch.save(feature_extractor.state_dict(), model_path)
                torch.save(classClassfier.state_dict(), model_path.replace("feacturer", "classer"))
def test():
    modelName = "D:/my bad/Suspicious object detection/Suspicious_Object_Detection/yue/fall_detect/models/hhh.pkl"
    dirname = "D:/my bad/Suspicious object detection/data/fall/notdx_cut300/1117_pre"
    k = 0  # 标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    for i in range(0, len(dataList)):
        path = os.path.join(dirname, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
            # print(temp_data.shape)
            if k == 0:
                raw_data = temp_data
                label = getlabel(dataList[i])
                k = 1
            else:
                raw_data = np.row_stack((raw_data, temp_data))
                label = np.row_stack((label, getlabel(dataList[i])))
    label = np_utils.to_categorical(label)
    print("raw_data:", raw_data.shape)
    print("label:", label.shape)
    data = raw_data
    label = label.astype(np.float32)

    test_feature = data
    print("test_feature" + str(test_feature.shape))

    # 全局归化为0~1
    a = test_feature.reshape(int(test_feature.shape[0] / sampleNum), sampleNum * 270)
    a = a.T
    min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    test_feature = min_max_scaler.fit_transform(a)
    print(test_feature.shape)
    test_feature = test_feature.T
    test_feature = test_feature.reshape([test_feature.shape[0], sampleNum, 270])
    test_feature = np.swapaxes(test_feature, 1, 2)
    print(test_feature.shape)


    feature_extractor = Feature_extractor(feature_len).to(device)
    classClassfier = ClassClassfier(class_num).to(device)
    feature_extractor.load_state_dict(torch.load(modelName.replace(".pkl", "feacturer.pkl")))
    classClassfier.load_state_dict(torch.load(modelName.replace(".pkl", "classer.pkl")))
    n = 0
    a_all = np.zeros((3, 2))  # 3个动作：第一个跌倒，后两个个非跌倒
    for o in range(4):  # 四个人的数据
        # print(test_feature.shape)
        non_mid =feature_extractor(torch.tensor(test_feature[o * 20:(o + 1) * 20], dtype=torch.float32).to(device))# 每个人20条数据，10条跌倒，10条非跌倒
        non_pre = classClassfier(non_mid)  # (20,2)
        m = 0
        a = np.zeros((3, 2))
        for i in range(4):
            for k in range(5):
                x = torch.argmax(non_pre[i * 5 + k])
                if (i == 0 or i == 1):
                    a[0][x] = a[0][x] + 1
                    a_all[0][x] = a_all[0][x] + 1
                else:
                    a[i - 1][x] = a[i - 1][x] + 1
                    a_all[i - 1][x] = a_all[i - 1][x] + 1
                if ((x == 0 and i <= 1) or (x == 1 and i >= 2)):
                    m = m + 1
                    n = n + 1
        acc = float(m) / float(len(non_pre))
        print("源" + str(o + 1) + "测试数据准确率：" + str(acc))
        print(a)
    ac = float(n) / float(80)
    k1 = ac
    print("源平均测试数据准确率：" + str(ac))
    print("精度：" + str(a_all[0:1, 0:1] / (a_all[0:1, 0:1] + a_all[1:2, 0:1] + a_all[2:3, 0:1])))
    print("召回率：" + str(a_all[0:1, 0:1] / (a_all[0:1, 0:1] + a_all[0:1, 1:2])))
    print(a_all)
#train()
test()







