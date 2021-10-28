import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
pre=np.arange(8*2)
pre=pre.reshape(2,8)
print(pre)
pre=pre.reshape(8,2)
print(pre)
print(pre.size())


min_max_scaler = MinMaxScaler(feature_range=[0,1])
pre=min_max_scaler.fit_transform(pre)
print(pre)
# pre1=np.arange(5*2)
# pre1=pre1.reshape(2,5)
# print(pre1)
# result=np.matmul(pre, pre1)
# print(result)


# pre=np.arange(5*2)
# pre=pre.reshape(5,2)
# #pre=pre.T
# print(pre)
# pre1=np.arange(5)
# pre1=pre1.reshape(5,1)
# #pre1=pre1.T
# print(pre1)
# result=pre*pre1
# #result=np.multiply(pre,pre1)
# print(result)
# print(pre+result)
# pre=np.zeros((2,5))
# print(ar1)
# print(ar1.shape)
#
# ar1=ar1.reshape(4,4)
# print(ar1)
# print(ar1.shape)
#
# ar1=ar1.reshape(2,2,4)
# print(ar1)
# print(ar1.shape)

# m=0
# for i in range(24):
#     for k in range(80):
#         if(np.argmax(pre[i*80+k])==23):
#             print(np.argmax(pre[i*80+k]))
#             m=m+1
# acc = float(m) / float(len(pre))
# print(acc)

# file = r'D:\my bad\Suspicious object detection\Suspicious_Object_Detection\yue\dingwei\test.txt'
# f = open(file,"ab+")    #可读可写二进制，文件若不存在就创建
# str='asfasgasg\n'
# f.write(str.encode())
# f.close() #关闭文件
# fn = 'D:/my bad/Suspicious object detection/data/huodong/room1/data.csv'
# csvdata = pd.read_csv(fn, header=None)
# csvdata = np.array(csvdata, dtype=np.float64)
# print(csvdata.shape)