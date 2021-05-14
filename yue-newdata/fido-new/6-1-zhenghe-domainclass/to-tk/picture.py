import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from filter import hampel, smooth, Kalman1D, savgol
from data_deal import peakRemoval

sampling_rate = 20  # 采样频率
data1 = pd.read_csv('/home/zhw/实验部分/危险品/1-2.csv')
data2 = pd.read_csv('/home/zhw/实验部分/危险品/2-2.csv')
data3 = pd.read_csv('/home/zhw/实验部分/危险品/3-2.csv')
data4 = pd.read_csv('/home/zhw/实验部分/危险品/4-2.csv')
# data3 = pd.read_csv('/home/zhw/实验部分/危险品/zhw-M-0.csv')w
print(len(data1))
print(len(data2))
# print(len(data3))
data1 = data1.iloc[:, 120:150]  # 120～150是中间一对收发器的30个子载波数据23000:23100
data2 = data2.iloc[:, 120:150]  # 120～150是中间一对收发器的30个子载波数据23000:23100
data3 = data3.iloc[:, 120:150]  # 120～150是中间一对收发器的30个子载波数据23000:23100
data4 = data4.iloc[:, 120:150]  # 120～150是中间一对收发器的30个子载波数据23000:23100
# data3 = data3.iloc[:, 120:150]  # 120～150是中间一对收发器的30个子载波数据23000:23100

t = range(200)

dataMatrix1 = data1.values
dataMatrix2 = data2.values
dataMatrix3 = data3.values
dataMatrix4 = data4.values
# dataMatrix3 = data3.values

# for x in range(2, 3):
#     filtedData = hampel(dataMatrix[:, x])  # 首先经过hampel滤波
#     filtedData = smooth(filtedData, 25)  # 经过滑动平均滤波，去除高频噪声
#     filtedData = savgol(filtedData, 5, 3, 1)  # 这是S-G滤波，其为滑动滤波的升级版，画图可见其去除高频噪声效果更好
#     filtedData=np.array(filtedData)
#     plt.plot(t[:], filtedData[:])

for x in range(3,9):
    filtedData1 = hampel(dataMatrix1[0:200, x])  # 首先经过hampel滤波
    filtedData1 = smooth(filtedData1, 5)  # 经过滑动平均滤波，去除高频噪声

    filtedData2 = hampel(dataMatrix2[0:200, x])  # 首先经过hampel滤波
    filtedData2 = smooth(filtedData2, 5)  # 经过滑动平均滤波，去除高频噪声

    filtedData3 = hampel(dataMatrix3[0:200, x])  # 首先经过hampel滤波
    filtedData3 = smooth(filtedData3, 5)  # 经过滑动平均滤波，去除高频噪声

    filtedData4 = hampel(dataMatrix4[0:200, x])  # 首先经过hampel滤波
    filtedData4 = smooth(filtedData4, 5)  # 经过滑动平均滤波，去除高频噪声

    plt.plot(t[:], filtedData1, 'r')
    plt.plot(t[:], filtedData2, 'g')
    plt.plot(t[:], filtedData3, 'b')
    plt.plot(t[:], filtedData4, 'y')


plt.show()
