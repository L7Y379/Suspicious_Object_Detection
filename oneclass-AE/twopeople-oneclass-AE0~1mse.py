#在测试阶段，如果一个样本是之前训练过的同类数据，那么它的重构错误很低。而另一个类型的重构错误较高。

import tensorflow as tf
from keras.layers import Dense, Input
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
    testfile = []
    for j in ["0", "1M"]:  # "1S", "2S"
        for name in ['zb','ljy']:
            for i in [i for i in range(0, 30)]:
                fn = filepath + name+"-2.5-M/" + name+"-"+ str(j) + "-" + str(i) + filetype
                filenames += [fn]
            np.random.shuffle(filenames)
            trainfile += filenames[:20]
            testfile += filenames[20:]
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
        for i in [i for i in range(0, 25)]:
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
tk_files=file_array_other()
train_feature, train_label = read_data(trainfile_array)
test_feature, test_label = read_data(testfile_array)
tk_feature,tk_label=read_data(tk_files)

#全局归一化
# train_feature = train_feature.astype('float32')/np.max(train_feature)
# test_feature = test_feature.astype('float32')/np.max(test_feature)
# tk_feature=tk_feature.astype('float32')/np.max(tk_feature)
train_feature = (train_feature.astype('float32')-np.min(np.concatenate((train_feature, test_feature), axis=0)))/(np.max(np.concatenate((train_feature, test_feature), axis=0))-np.min(np.concatenate((train_feature, test_feature), axis=0)))
test_feature = (test_feature.astype('float32')-np.min(np.concatenate((train_feature, test_feature), axis=0)))/(np.max(np.concatenate((train_feature, test_feature), axis=0))-np.min(np.concatenate((train_feature, test_feature), axis=0)))
tk_feature=(tk_feature.astype('float32')-np.min(np.concatenate((train_feature, test_feature), axis=0)))/(np.max(np.concatenate((train_feature, test_feature), axis=0))-np.min(np.concatenate((train_feature, test_feature), axis=0)))
# train_feature = train_feature.astype('float32')/73.0
# test_feature = test_feature.astype('float32')/73.0
# tk_feature=tk_feature.astype('float32')/73.0

# #列归一化
# min_max_scaler = preprocessing.MinMaxScaler()
# train_feature = min_max_scaler.fit_transform(train_feature)
# test_feature = min_max_scaler.fit_transform(test_feature)
# tk_feature = min_max_scaler.fit_transform(tk_feature)

# #行归一化
# min_max_scaler1 = preprocessing.MinMaxScaler()
# train_feature=train_feature.T
# test_feature=test_feature.T
# tk_feature=tk_feature.T
# train_feature = min_max_scaler1.fit_transform(train_feature)
# test_feature = min_max_scaler1.fit_transform(test_feature)
# tk_feature = min_max_scaler1.fit_transform(tk_feature)
# train_feature=train_feature.T
# test_feature=test_feature.T
# tk_feature=tk_feature.T

train_feature_nosiy = train_feature
test_feature_nosiy = test_feature
# train_feature_nosiy = np.clip(train_feature_nosiy, 0., 1.)
# test_feature_nosiy = np.clip(test_feature_nosiy, 0, 1.)
input = Input(shape=(270,))

encoded1 = Dense(128, activation='relu')(input)
# encoded1 = Dense(128, activation='relu')(encoded1)
encoded2 = Dense(64,activation='relu')(encoded1)
decoded1 = Dense(128, activation='relu')(encoded2)
# decoded1 = Dense(128, activation='relu')(decoded1)
#decoded1 = Dense(128, activation='relu')(decoded1)
#decoded2 = Dense(270, activation='sigmoid')(decoded1)
decoded2 = Dense(270, activation='sigmoid')(decoded1)
#decoded2 = Dense(270, activation='relu')(decoded1)

autoencoder = Model(input=input, output=decoded2)
#print(autoencoder.inputs)
autoencoder_mid = Model(inputs=input, outputs=encoded2)

autoencoder.compile(optimizer='adam', loss='mse')
#autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()
autoencoder.fit(train_feature_nosiy[:9600], train_feature[:9600], epochs=200, batch_size=128, verbose=1, validation_data=(test_feature_nosiy[:4800], test_feature[:4800]))

#decoded test images
train_predict = autoencoder.predict(train_feature_nosiy)
test_predict = autoencoder.predict(test_feature_nosiy)
tk_predict = autoencoder.predict(tk_feature)
print(train_predict)
print(test_predict)
sess = tf.Session()
# Evaluate the tensor `c`.

train_loss = tf.reduce_mean(tf.square(train_feature_nosiy - train_predict),axis=1)
print(train_loss)
print(sess.run(train_loss))
print(sess.run(train_loss[:100]))
print(sess.run(train_loss[4700:]))

test_loss = tf.reduce_mean(tf.square(test_feature_nosiy - test_predict),axis=1)
print(test_loss)
print(sess.run(test_loss))
print(sess.run(test_loss[:100]))
print(sess.run(test_loss[2300:]))

tk_loss= tf.reduce_mean(tf.square(tk_feature - tk_predict),axis=1)

#设置经验基准m
len_train=train_loss.shape[0]
print(len_train)
train_loss=train_loss.eval(session=sess)#转换为数组
# m=(np.max(train_loss[:2400])+np.min(train_loss[2400:]))/2#取前后两部分最大和最小值的均值
# m=np.mean(train_loss)#取平均值
m=np.median(train_loss)#取中位数
print("m为")
print(m)
pred_train=np.arange(int(len_train))
for i in range(0,int(len_train)):
    if train_loss[i]<=m:pred_train[i]=0
    if train_loss[i]>m:pred_train[i]=1
len_test=test_loss.shape[0]
test_loss=test_loss.eval(session=sess)#转换为数组
pred_test=np.arange(int(len_test))
for i in range(0,int(len_test)):
    if test_loss[i]<=m:pred_test[i]=0
    if test_loss[i]>m:pred_test[i]=1
len_tk=tk_loss.shape[0]
tk_loss=tk_loss.eval(session=sess)#转换为数组
pred_tk=np.arange(int(len_tk))
for i in range(0,int(len_tk)):
    if tk_loss[i]<=m:pred_tk[i]=0
    if tk_loss[i]>m:pred_tk[i]=1
#
a1=[0,0]
a2=[0,0]
for i in range(0,int(len(pred_train)/2)):
    if pred_train[i]==0:a1[0]=a1[0]+1
    if pred_train[i]==1:a1[1]=a1[1]+1
for j in range(int(len(pred_train)/2),int(len(pred_train))):
    if pred_train[j] == 0: a2[0] = a2[0] + 1
    if pred_train[j] == 1: a2[1] = a2[1] + 1
print(a1)
print("没带东西正确判定的准确率为：", end='')
print(float(a1[0])/float(a1[0]+a1[1]))
print(a2)
print("带了东西正确判定的准确率为：", end='')
print(float(a2[1])/float(a2[0]+a2[1]))
if((a1[0]+a2[1])>=(a1[1]+a2[0])):
    a=(a1[0]+a2[1])
    c=0
else:
    a=(a1[1]+a2[0])
    c=1
acc_train=float(a)/float(len(pred_train))
print("训练数据总体准确率为：")
print(acc_train)
#print(c)
b1 = [0, 0]
b2 = [0, 0]
for i in range(0, int(len(pred_test) / 2)):
    if pred_test[i] == 0:b1[0] = b1[0] + 1
    if pred_test[i] == 1:b1[1] = b1[1] + 1
for j in range(int(len(pred_test) / 2), int(len(pred_test))):
    if pred_test[j] == 0:b2[0] = b2[0] + 1
    if pred_test[j] == 1:b2[1] = b2[1] + 1
print(b1)
print("没带东西正确判定的准确率为：", end='')
print(float(b1[0])/float(b1[0]+b1[1]))
print(b2)
print("带了东西正确判定的准确率为：", end='')
print(float(b2[1])/float(b2[0]+b2[1]))
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
print("测试数据总体准确率为：")
print(acc_test)

#投票
def get_max(shuzu):
    s=[0,0]
    for i in range(0,lin2):
        if (shuzu[i]==0):s[0]=s[0]+1
        else:s[1]=s[1]+1
    if(s[0]>s[1]):return 0
    if(s[0]<s[1]):return 1
    if(s[0]==s[1]):return 2

pred_train_vot=np.arange(len(pred_train)/lin2)
print(len(pred_train_vot))
for b in range(0, len(pred_train_vot)):
    i=get_max(pred_train[b*lin2:(b+1)*lin2])
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
print("投票后没带东西正确判定的准确率为：", end='')
print(float(a1[0])/float(a1[0]+a1[1]))
print(a2)
print("投票后带了东西正确判定的准确率为：", end='')
print(float(a2[1])/float(a2[0]+a2[1]))
if((a1[0]+a2[1])>=(a1[1]+a2[0])):
    a=(a1[0]+a2[1])
    c=0
else:
    a=(a1[1]+a2[0])
    c=1
acc_train_vot=float(a)/float(len(pred_train_vot))
print("训练数据的投票后准确率为：")
print(acc_train_vot)

pred_test_vot = np.arange(len(pred_test) / lin2)
print(len(pred_test_vot))
for b in range(0, len(pred_test_vot)):
    i = get_max(pred_test[b * lin2:(b + 1) * lin2])
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
print("投票后没带东西正确判定的准确率为：", end='')
print(float(b1[0])/float(b1[0]+b1[1]))
print(b2)
print("投票后带了东西正确判定的准确率为：", end='')
print(float(b2[1])/float(b2[0]+b2[1]))
if((b1[0]+b2[1])>=(b1[1]+b2[0])):
    if (c==0):
        b=(b1[0]+b2[1])
    else:b = (b1[1] + b2[0])
else:
    if(c==1):
        b = (b1[1] + b2[0])
    else:b=(b1[0]+b2[1])
acc_test_vot = float(b) / float(len(pred_test_vot))
print("测试数据的投票后准确率为：")
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

pred_tk_vot = np.arange(len(pred_tk) / lin2)
print(len(pred_tk_vot))
for b in range(0, len(pred_tk_vot)):
    i = get_max(pred_tk[b * lin2:(b + 1) * lin2])
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
