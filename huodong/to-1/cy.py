#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torchvision
import torch.nn.functional as F
import glob as glob
import pandas as pd
import numpy as np
import torch.optim as optim
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.autograd import Function
import math
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
import argparse
from torch.autograd import Function
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch import nn
from torchvision.utils import save_image
import random
from torch import nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[2]:


files = ['E:\\JupyterProjrcts\\CY\\Dataset\\5L_cross_position\\data.csv', 'E:\\JupyterProjrcts\\CY\\Dataset\\5L_cross_position\\label.csv']
for file in files:
    read_data = pd.read_csv(file, header=None, skip_blank_lines=True, dtype=np.float32)
    if file.find("label") == -1:
        gesture_data = read_data.values
    else:
        gesture_label = read_data.values


# In[13]:


plt.plot(gesture_data[0][::270])
ddd = []
for i in  gesture_data[0][::270]:
    ddd.append(i)
ddd


# In[17]:


gesture_data[0][:500]


# In[18]:


index = []
uuu = ""
for i in range(300):
    index.append(i)
    uuu += ",{}".format(i)
uuu


# In[12]:


scaler = MinMaxScaler(feature_range=(0,1), copy=False)
scaler_gesture_data = np.array(scaler.fit_transform(gesture_data))
scaler_gesture_data = gesture_data.reshape([-1, 200, 270]).transpose(0,2,1)
print(scaler_gesture_data.shape)


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
          nn.Conv1d(in_channels = feature_len, out_channels=200, kernel_size=20),
          nn.BatchNorm1d(200),
          nn.ReLU(),

          nn.Conv1d(in_channels = 200, out_channels=150, kernel_size=20),
          nn.BatchNorm1d(150),
          nn.ReLU(),

          nn.Conv1d(in_channels = 150, out_channels=100, kernel_size=20),
          nn.BatchNorm1d(100),
          nn.ReLU(),

          nn.Conv1d(in_channels = 100, out_channels=50, kernel_size=20),
          nn.BatchNorm1d(50),
          nn.ReLU(),

          nn.Conv1d(in_channels = 50, out_channels=10, kernel_size=20),
          nn.Flatten()
        )

    def forward(self, ipt):
        feature = self.feature_extractor(ipt)

        return feature


class ClassClassfier(nn.Module):
    def __init__(self, class_num):
        super(ClassClassfier, self).__init__()
        self.class_classifier = nn. Sequential(
          nn.Linear(1050, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Linear(64, class_num),
          nn.Softmax()
        )

    def forward(self, ipt):
        class_out = self.class_classifier(ipt)

        return class_out



# In[40]:


def get_acc(all_loders, target_loder):
    lr = 0.001
    class_num = 6
    feature_len = 270
    epochs = 100
    time_seq = 200

    feature_extractor = Feature_extractor(feature_len).to(device)
    classClassfier = ClassClassfier(class_num).to(device)

    loss_class = nn.CrossEntropyLoss()

    optimizer_feature_extractor = optim.Adam(feature_extractor.parameters(), lr)
    optimizer_classClassfier = optim.Adam(classClassfier.parameters(), lr)
    loder_len = len(target_loder)


    for epoch in range(1, 1+epochs):
        all_iters = []
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        for i in range(4):
            all_iters.append(iter(all_loders[i]))
        for iter_num in range(loder_len):
            for i in range(4):
                souce_item = all_iters[i].next()
                mmd_loss = torch.tensor(0.0)
                mmd_loss = Variable(mmd_loss, requires_grad=True)
                source_item_csi_data,  source_item_label = souce_item[0].to(device), souce_item[1].to(device)
                source_item_label = torch.argmax(source_item_label,1)

                optimizer_feature_extractor.zero_grad()
                optimizer_classClassfier.zero_grad()

                feature = feature_extractor(source_item_csi_data)
                class_out = classClassfier(feature)
                if i == 0:
                    last_feature = feature.detach()
                    last_label = source_item_label.detach()
                else:
                    mmd_loss = cal_mmd(last_feature, feature, last_label, source_item_label)
#                     mmd_loss = cal_mmd(last_feature, feature)

                loss = loss_class(class_out, source_item_label) + mmd_loss*lambd
                loss.backward()

                optimizer_feature_extractor.step()
                optimizer_classClassfier.step()

        # print(loss)

        acc_num = 0
        total_num = 0
        for data in target_loder:
            t_img, t_label = data[0].to(device), data[1].to(device)
            t_label = torch.argmax(t_label,1)

            feature = feature_extractor(t_img)
            pred = classClassfier(feature)
            pred = torch.argmax(pred, 1)

            acc_num += (pred == t_label).sum().item()
            total_num += t_img.shape[0]

        print('t-acc:', acc_num / total_num)
    return feature_extractor


# In[ ]:


stack_scaler_gesture_data = np.vstack((scaler_gesture_data,scaler_gesture_data))
stack_gesture_label = np.vstack((gesture_label,gesture_label))
batchsz =  60
all_feature_extractors = []

for k in range(1):
    all_loders = []
    start = k*120
    end = k * 120 + 120
    for i in range(4):
        train_data = torch.from_numpy(stack_scaler_gesture_data[end+i*120:end+i*120+120])
        train_label = torch.from_numpy(stack_gesture_label[end+i*120:end+i*120+120]).type(torch.long)
        source_set = torch.utils.data.TensorDataset(train_data, train_label)
        source_loder = torch.utils.data.DataLoader(source_set, batch_size = batchsz, shuffle=True)

        all_loders.append(source_loder)

    target_data = torch.from_numpy(stack_scaler_gesture_data[start:end])
    target_label = torch.from_numpy(stack_gesture_label[start:end]).type(torch.long)
    target_set = torch.utils.data.TensorDataset(target_data, target_label)
    target_loder = torch.utils.data.DataLoader(target_set, batch_size = batchsz, shuffle=True)

    print(k+1)
    all_feature_extractors.append(get_acc(all_loders, target_loder))


# In[20]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2, random_state=0)


# In[25]:


feature_extractor = all_feature_extractors[0]
total_source_data = torch.from_numpy(scaler_gesture_data[:480])
total_source_label = torch.from_numpy(gesture_label[:480]).type(torch.long)
total_source_label = torch.argmax(total_source_label,1)
total_feature = feature_extractor(total_source_data.to(device)).cpu().detach().numpy()
tsne_res = tsne.fit_transform(total_feature)
sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = total_source_label, palette = sns.hls_palette(6), legend = 'full');


# In[37]:


feature_extractor = all_feature_extractors[0]
total_source_data = torch.from_numpy(scaler_gesture_data[120:])
total_source_label = torch.from_numpy(gesture_label[120:]).type(torch.long)
total_source_label = torch.argmax(total_source_label,1)
total_feature = feature_extractor(total_source_data.to(device)).cpu().detach().numpy()
tsne_res = tsne.fit_transform(total_feature)
sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = total_source_label, palette = sns.hls_palette(6), legend = 'full');


# In[ ]:


feature_extractor = all_feature_extractors[0]
total_source_data = torch.from_numpy(scaler_gesture_data[120:])
total_source_label = torch.from_numpy(gesture_label[120:]).type(torch.long)
total_source_label = torch.argmax(total_source_label,1)
total_feature = feature_extractor(total_source_data.to(device)).cpu().detach().numpy()
tsne_res = tsne.fit_transform(total_feature)
sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = total_source_label, palette = sns.hls_palette(6), legend = 'full');


# In[ ]:




