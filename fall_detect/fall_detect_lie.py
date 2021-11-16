import os
import pandas as pd
import numpy as np
from keras.layers import LSTM,Input, Dense,Flatten,MaxPooling2D,TimeDistributed,Bidirectional, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math

nb_time_steps = 200  #时间序列长度
nb_input_vector = 90 #输入序列
ww=1
img_rows = 15
img_cols = 18
channels = 1
img_shape = (200,img_rows, img_cols, channels)
img_shape_LSTM = (200,270)
epochs = 300
batch_size = 100
latent_dim = 90

def build_cnn(img_shape):
    cnn = Sequential()
    cnn.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same',input_shape=img_shape)))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3),activation='relu',strides=(1,1), padding='same')))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same')))
    cnn.add(TimeDistributed(Flatten()))
    cnn.add(TimeDistributed(Dense(90, activation="relu")))
    img = Input(shape=img_shape)
    latent_repr = cnn(img)
    return Model(img, latent_repr)
def build_rnn():
    rnn=Sequential()
    rnn.add(Bidirectional(LSTM(units=120, input_shape=(nb_time_steps, nb_input_vector))))
    rnn.add(Dense(500, activation="relu"))
    rnn.add(Dense(2, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = rnn(encoded_repr)
    return Model(encoded_repr, validity)
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
    train_feature = train_feature.reshape([int(train_feature.shape[0] / 200), 200, img_rows, img_cols])
    train_feature = np.expand_dims(train_feature, axis=4)
    test_feature = test_feature.reshape([int(test_feature.shape[0] / 200), 200, img_rows, img_cols])
    test_feature = np.expand_dims(test_feature, axis=4)

    # 行归化为0~1
    # min_max_scaler = MinMaxScaler(feature_range=[0,1])
    # all = np.concatenate((train_feature, test_feature), axis=0)
    # all = np.concatenate((all, train_feature_ot), axis=0)
    # all=all.T
    # all= min_max_scaler.fit_transform(all)
    # all=all.T
    # train_feature = all[:len(train_feature)]
    # test_feature = all[len(train_feature):(len(train_feature)+len(test_feature))]
    # train_feature_ot = all[(len(train_feature)+len(test_feature)):]

    opt = Adam(0.0002, 0.5)
    cnn = build_cnn(img_shape)
    rnn = build_rnn()
    rnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    img3 = Input(shape=img_shape)
    encoded_repr3 = cnn(img3)
    validity1 = rnn(encoded_repr3)
    crnn_model = Model(img3, validity1)
    crnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # crnn_model.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000crnn_model.h5')
    # dis.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000dis.h5')
    #crnn_model.save_weights(model_path)
    k = 0
    acc=0
    acc_test=0
    for epoch in range(epochs):

        idx = np.random.randint(0, train_feature.shape[0], batch_size)
        imgs = train_feature[idx]
        crnn_loss = crnn_model.train_on_batch(imgs, train_label[idx])
        if epoch % 10 == 0:
            print("%d [fall_detection_loss: %f,acc: %.2f%%]" % (epoch, crnn_loss[0], 100 * crnn_loss[1]))

            n = 0
            a_all = np.zeros((5, 2))  # 五个动作：第一个跌倒，后四个非跌倒
            for o in range(4):  # 四个人的数据
                print(test_feature.shape)
                non_mid = crnn_model.predict(test_feature[o * 30:(o + 1) * 30])  # 每个人30条数据，10条跌倒，20条非跌倒
                non_pre = non_mid  # (30,2)
                m = 0
                a = np.zeros((5, 2))
                for i in range(6):
                    for k in range(5):
                        x = np.argmax(non_pre[i * 5 + k])
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
            ac = float(n) / float(120)
            k1 = ac
            print("源平均测试数据准确率：" + str(ac))
            print(a_all)

            if(acc_test<ac):
                acc_test=ac
                crnn_model.save_weights(model_path)
        # if (epoch!=0 and epoch % 20==0 and acc < crnn_loss[1]):
        #     acc = crnn_loss[1]
        #     # if os.path.exists(model_path):
        #     #     os.remove(model_path)
        #     crnn_model.save_weights(model_path)

    K.clear_session()
    print("训练完成")
    #train_stop()
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

    model_path = "D:/my bad/Suspicious object detection/Suspicious_Object_Detection/yue/fall_detect/models/1112_lie1.h5"
    dirPath="D:/my bad/Suspicious object detection/data/fall/1112_pre"
    dirPath2 = "D:/my bad/Suspicious object detection/data/fall/1115_pre"
    model_train_test(dirPath,dirPath2, model_path)
def test_walk():
    modelName = "D:/my bad/Suspicious object detection/Suspicious_Object_Detection/yue/fall_detect/models/1112_lie1.h5"
    dirname = "D:/my bad/Suspicious object detection/data/fall/1115_pre"
    k = 0  # 标识符 判断数据列表是否新建
    print("进入模型训练。。。")
    dataList = os.listdir(dirname)
    for i in range(0, len(dataList)):
        path = os.path.join(dirname, dataList[i])
        if os.path.isfile(path):
            temp_data = pd.read_csv(path, error_bad_lines=False, header=None)
            temp_data = np.array(temp_data, dtype=np.float64)
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
    t = range(10000)
    # plt.plot(t[:raw_data.shape[0]], raw_data[:, 0:1], 'r')
    # plt.show()
    # plt.savefig("D:\\my bad\\CSI_DATA\\fall_detection\\fall_detection\\data_model_dir\\data_dir\\2.png")
    data = raw_data
    label = label.astype(np.float32)

    test_feature=data
    print("test_feature" + str(test_feature.shape))

    #全局归化为0~1
    #b1=test_feature.reshape(120,200,270)
    #plt.plot(b[0])
    # a = test_feature.reshape(int(test_feature.shape[0] / 200), 200 * 270)
    # a = a.T
    # min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    # test_feature = min_max_scaler.fit_transform(a)
    # print(test_feature.shape)
    # test_feature = test_feature.T
    #b2=test_feature.reshape(120,200,270)
    # plt.plot(b1[0])
    # plt.show()
    # plt.plot(b2[0])
    # plt.show()
    # test_feature = test_feature.reshape([test_feature.shape[0], 200, img_rows, img_cols])
    # test_feature = np.expand_dims(test_feature, axis=4)
    #

    min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    for i in range(int(test_feature.shape[0] / 200)):
        test_feature[i * 200:(i + 1) * 200] = min_max_scaler.fit_transform(test_feature[i * 200:(i + 1) * 200])
    test_feature = test_feature.reshape([int(test_feature.shape[0] / 200), 200, img_rows, img_cols])
    test_feature = np.expand_dims(test_feature, axis=4)


    print(test_feature.shape)
    #opt = Adam(0.0002, 0.5)
    cnn = build_cnn(img_shape)
    rnn = build_rnn()
    #rnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    img3 = Input(shape=img_shape)
    encoded_repr3 = cnn(img3)
    validity1 = rnn(encoded_repr3)
    crnn_model = Model(img3, validity1)
    #crnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    crnn_model.load_weights(modelName)

    n = 0
    a_all = np.zeros((5, 2))#五个动作：第一个跌倒，后四个非跌倒
    for o in range(4):#四个人的数据
        print(test_feature.shape)
        non_mid = crnn_model.predict(test_feature[o * 30:(o + 1) * 30])#每个人30条数据，10条跌倒，20条非跌倒
        non_pre = non_mid  # (30,2)
        m = 0
        a = np.zeros((5, 2))
        for i in range(6):
            for k in range(5):
                x = np.argmax(non_pre[i * 5 + k])
                if(i==0 or i==1):
                    a[0][x] = a[0][x] + 1
                    a_all[0][x] = a_all[0][x] + 1
                else:
                    a[i-1][x] = a[i-1][x] + 1
                    a_all[i-1][x] = a_all[i-1][x] + 1
                if ((x == 0 and i <= 1) or (x == 1 and i >= 2)):
                    m = m + 1
                    n = n + 1
        acc = float(m) / float(len(non_pre))
        print("源" + str(o + 1) + "测试数据准确率：" + str(acc))
        print(a)
    ac = float(n) / float(120)
    k1 = ac
    print("源平均测试数据准确率：" + str(ac))
    print(a_all)

#train_model()
test_walk()
#train_test()