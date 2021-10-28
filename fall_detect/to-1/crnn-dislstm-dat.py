#crnn dislstm
import pandas as pd
import os
from sklearn.cluster import KMeans
from keras.layers import LSTM,Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D,TimeDistributed
from keras.layers import Lambda,Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from keras.utils import np_utils
import time
from sklearn.preprocessing import MinMaxScaler
import math

def read_bf_file(filename, decoder="python"):
    with open(filename, "rb") as f:
        bfee_list = []
        field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
        while field_len != 0:
            bfee_list.append(f.read(field_len))
            field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)

    dicts = []

    count = 0  # % Number of records output
    broken_perm = 0  # % Flag marking whether we've encountered a broken CSI yet
    triangle = [0, 1, 3]  # % What perm should sum to for 1,2,3 antennas

    csi_len = len(bfee_list)
    for array in bfee_list[:(csi_len - 1)]:
        # % Read size and code
        code = array[0]

        # there is CSI in field if code == 187，If unhandled code skip (seek over) the record and continue
        if code != 187:
            # % skip all other info
            continue
        else:
            # get beamforming or phy data
            count = count + 1

            timestamp_low = int.from_bytes(array[1:5], byteorder='little', signed=False)
            bfee_count = int.from_bytes(array[5:7], byteorder='little', signed=False)
            Nrx = array[9]
            Ntx = array[10]
            rssi_a = array[11]
            rssi_b = array[12]
            rssi_c = array[13]
            noise = array[14] - 256
            agc = array[15]
            antenna_sel = array[16]
            b_len = int.from_bytes(array[17:19], byteorder='little', signed=False)
            fake_rate_n_flags = int.from_bytes(array[19:21], byteorder='little', signed=False)
            payload = array[21:]  # get payload

            calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 6) / 8
            perm = [1, 2, 3]
            perm[0] = ((antenna_sel) & 0x3)
            perm[1] = ((antenna_sel >> 2) & 0x3)
            perm[2] = ((antenna_sel >> 4) & 0x3)

            # Check that length matches what it should
            if (b_len != calc_len):
                print("MIMOToolbox:read_bfee_new:size", "Wrong beamforming matrix size.")

            # Compute CSI from all this crap :
            if decoder == "python":
                csi = parse_csi(payload, Ntx, Nrx)
            else:
                csi = None
                print("decoder name error! Wrong encoder name:", decoder)
                return

            # % matrix does not contain default values
            if sum(perm) != triangle[Nrx - 1]:
                print('WARN ONCE: Found CSI (', filename, ') with Nrx=', Nrx, ' and invalid perm=[', perm, ']\n')
            else:
                csi[:, perm, :] = csi[:, [0, 1, 2], :]

            # dict,and return
            bfee_dict = {
                'timestamp_low': timestamp_low,
                'bfee_count': bfee_count,
                'Nrx': Nrx,
                'Ntx': Ntx,
                'rssi_a': rssi_a,
                'rssi_b': rssi_b,
                'rssi_c': rssi_c,
                'noise': noise,
                'agc': agc,
                'antenna_sel': antenna_sel,
                'perm': perm,
                'len': b_len,
                'fake_rate_n_flags': fake_rate_n_flags,
                'calc_len': calc_len,
                'csi': csi}

            dicts.append(bfee_dict)

    return dicts
def parse_csi_new(payload, Ntx, Nrx):
    # Compute CSI from all this crap
    csi = np.zeros(shape=(30, Nrx, Ntx), dtype=np.dtype(np.complex))
    index = 0

    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                real_bin = (int.from_bytes(payload[int(index / 8):int(index / 8 + 2)], byteorder='big',
                                           signed=True) >> remainder) & 0b11111111
                real = real_bin
                imag_bin = bytes([(payload[int(index / 8 + 1)] >> remainder) | (
                            payload[int(index / 8 + 2)] << (8 - remainder)) & 0b11111111])
                imag = int.from_bytes(imag_bin, byteorder='little', signed=True)
                tmp = np.complex(float(real), float(imag))
                csi[i, j, k] = tmp
                index += 16
    return csi
def parse_csi(payload, Ntx, Nrx):
    # Compute CSI from all this crap
    csi = np.zeros(shape=(Ntx, Nrx, 30), dtype=np.dtype(np.complex))
    index = 0

    for i in range(30):
        index += 3
        remainder = index % 8
        for j in range(Nrx):
            for k in range(Ntx):
                start = index // 8
                real_bin = bytes([(payload[start] >> remainder) | (payload[start + 1] << (8 - remainder)) & 0b11111111])
                real = int.from_bytes(real_bin, byteorder='little', signed=True)
                imag_bin = bytes(
                    [(payload[start + 1] >> remainder) | (payload[start + 2] << (8 - remainder)) & 0b11111111])
                imag = int.from_bytes(imag_bin, byteorder='little', signed=True)
                tmp = np.complex(float(real), float(imag))
                csi[k, j, i] = tmp
                index += 16
    return csi
def db(X, U):
    R = 1
    if 'power'.startswith(U):
        assert X >= 0
    else:
        X = math.pow(abs(X), 2) / R

    return (10 * math.log10(X) + 300) - 300
def dbinv(x):
    return math.pow(10, x / 10)
def get_total_rss(csi_st):
    # Careful here: rssis could be zero
    rssi_mag = 0
    if csi_st['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_a'])
    if csi_st['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_b'])
    if csi_st['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_c'])
    return db(rssi_mag, 'power') - 44 - csi_st['agc']
def get_scale_csi(csi_st):
    # Pull out csi
    csi = csi_st['csi']
    # print(csi.shape)
    # print(csi)
    # Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = np.multiply(csi, np.conj(csi)).real
    csi_pwr = np.sum(csi_sq, axis=0)
    csi_pwr = csi_pwr.reshape(1, csi_pwr.shape[0], -1)
    rssi_pwr = dbinv(get_total_rss(csi_st))

    csi_pwr[csi_pwr==0] = 1
    scale = rssi_pwr / (csi_pwr / 30)

    if csi_st['noise'] == -127:
        noise_db = -92
    else:
        noise_db = csi_st['noise']
    thermal_noise_pwr = dbinv(noise_db)

    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])

    total_noise_pwr = thermal_noise_pwr + quant_error_pwr
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st['Ntx'] == 2:
        ret = ret * math.sqrt(2)
    elif csi_st['Ntx'] == 3:
        ret = ret * math.sqrt(dbinv(4.5))
    return ret
localtime1 = time.asctime( time.localtime(time.time()) )
print ("本地时间为 :", localtime1)
#nb_lstm_outputs = 80  #神经元个数
nb_time_steps = 200  #时间序列长度
nb_input_vector = 90 #输入序列
ww=1
img_rows = 9
img_cols = 10
channels = 1
img_shape = (200,img_rows, img_cols, channels)
epochs = 5000
batch_size = 100
latent_dim = 90
def read_datamid_huodong():
    k=1
    fn1 = 'D:/my bad/Suspicious object detection/data/huodong/room1/data.csv'
    csvdata = pd.read_csv(fn1, header=None)
    csvdata = np.array(csvdata, dtype=np.float64)
    print(csvdata.shape)  # (1200,18000)
    train_feature = csvdata.reshape(240000, 90)
    train_feature_ot = train_feature[(k - 1) * 120 * 200:k * 120 * 200]
    train_feature = np.concatenate((train_feature[:(k - 1) * 120 * 200], train_feature[k * 120 * 200:]), axis=0)
    for i in range(54):
        idx1 = np.array([j for j in range((i * 20) * 200, (i * 20 + 8) * 200, 1)])  # 取中心点处左右分布数据
        idx2 = np.array([j for j in range((i * 20 + 13) * 200, (i * 20 + 20) * 200, 1)])  # 取中心点处左右分布数据
        idx = np.hstack((idx1, idx2))
        idxt = np.array([j for j in range((i * 20 + 8) * 200, (i * 20 + 13) * 200, 1)])
        # print(filename)
        if i == 0:
            temp_feature = train_feature[idx]
            temp_feature2 = train_feature[idxt]
        else:
            temp_feature = np.concatenate((temp_feature, train_feature[idx]), axis=0)
            temp_feature2 = np.concatenate((temp_feature2, train_feature[idxt]), axis=0)
    train_feature = temp_feature
    test_feature = temp_feature2
    print(train_feature.shape)  # (162000,90)
    print(test_feature.shape)  # (54000,90)
    print(train_feature_ot.shape)  # (24000,90)

    for i in range(9):
        temp_domain_label = np.tile(i, (90,))
        if i == 0:
            train_domain_label = temp_domain_label
        else:
            train_domain_label = np.concatenate((train_domain_label, temp_domain_label), axis=0)
    print(train_domain_label.shape)  # (162000,)
    train_domain_label = np_utils.to_categorical(train_domain_label)
    # train_domain_label = train_domain_label.reshape(810,200,9)
    print(train_domain_label.shape)  # (162000,9)
    print(train_domain_label.shape)


    temp_label = np.tile(0, (60,))
    temp_label2 = np.tile(1, (30,))
    temp_label=np.concatenate((temp_label, temp_label2), axis=0)
    temp_label_all=temp_label
    for i in range(8):
        temp_label_all = np.concatenate((temp_label_all, temp_label), axis=0)
    train_label=temp_label_all
    temp_label = np.tile(0, (20,))
    temp_label2 = np.tile(1, (10,))
    temp_label = np.concatenate((temp_label, temp_label2), axis=0)
    temp_label_all = temp_label
    for i in range(8):
        temp_label_all = np.concatenate((temp_label_all, temp_label), axis=0)
    test_label=temp_label_all

    temp_label = np.tile(0, (80,))
    temp_label2 = np.tile(1, (40,))
    temp_label = np.concatenate((temp_label, temp_label2), axis=0)
    train_label_ot=temp_label

    train_label=np_utils.to_categorical(train_label)
    test_label = np_utils.to_categorical(test_label)
    train_label_ot = np_utils.to_categorical(train_label_ot)

    print("train_label"+str(train_label.shape))  # (810,2)
    print("test_label"+str(test_label.shape))  # (270,2)
    print("train_label_ot"+str(train_label_ot.shape))  # (120,2)

    return np.array(train_feature), np.array(test_feature),np.array(train_feature_ot),np.array(train_domain_label),np.array(train_label),np.array(test_label),np.array(train_label_ot)

def build_cnn(img_shape):
    cnn = Sequential()
    cnn.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same',input_shape=img_shape)))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3),activation='relu',strides=(1,1), padding='same')))
    cnn.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),strides=(3,2))))
    cnn.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same')))
    cnn.add(TimeDistributed(Flatten()))
    cnn.add(TimeDistributed(Dense(180, activation="relu")))
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
def build_dis():
    dis = Sequential()
    dis.add(Bidirectional(LSTM(units=120, input_shape=(nb_time_steps, nb_input_vector))))
    dis.add(Dense(500, activation="relu"))
    dis.add(Dense(9, activation="softmax"))
    encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    validity = dis(encoded_repr)
    return Model(encoded_repr, validity)

def train():
    train_feature, test_feature, train_feature_ot, train_domain_label, train_label, test_label, train_label_ot = read_datamid_huodong()
    print("train_feature" + str(train_feature.shape))
    print("test_feature" + str(test_feature.shape))
    print("train_feature_ot" + str(train_feature_ot.shape))
    print("train_domain_label" + str(train_domain_label.shape))
    print("train_label" + str(train_label.shape))
    print("test_label" + str(test_label.shape))
    print("train_label_ot" + str(train_label_ot.shape))

    # 全局归化为0~1
    # a=np.concatenate((train_feature, test_feature), axis=0)
    # train_feature = (train_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
    # test_feature = (test_feature.astype('float32')-np.min(a))/(np.max(a)-np.min(a))
    # train_feature_ot=(train_feature_ot.astype('float32')-np.min(a))/(np.max(a)-np.min(a))

    # 列归化为0~1
    min_max_scaler = MinMaxScaler(feature_range=[0, 1])
    all = np.concatenate((train_feature, test_feature), axis=0)
    all = np.concatenate((all, train_feature_ot), axis=0)
    all = min_max_scaler.fit_transform(all)
    train_feature = all[:len(train_feature)]
    test_feature = all[len(train_feature):(len(train_feature) + len(test_feature))]
    train_feature_ot = all[(len(train_feature) + len(test_feature)):]

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

    train_feature = train_feature.reshape([int(train_feature.shape[0] / 200), 200, img_rows, img_cols])
    test_feature = test_feature.reshape([int(test_feature.shape[0] / 200), 200, img_rows, img_cols])
    train_feature_ot = train_feature_ot.reshape([int(train_feature_ot.shape[0] / 200), 200, img_rows, img_cols])
    train_feature = np.expand_dims(train_feature, axis=4)
    test_feature = np.expand_dims(test_feature, axis=4)
    train_feature_ot = np.expand_dims(train_feature_ot, axis=4)

    # def build_cnn(latent_dim2, img_shape):
    #     deterministic = 1
    #     img = Input(shape=img_shape)
    #     h = Flatten()(img)
    #     h = Dense(400, activation="relu")(h)
    #     h = Dense(400, activation="relu")(h)
    #     h = Dense(400, activation="relu")(h)
    #     latent_repr = Dense(latent_dim2)(h)
    #     return Model(img, latent_repr)
    # def build_cnn(latent_dim2, img_shape):
    #     model = Sequential()
    #     model.add(Conv2D(8, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same',input_shape=img_shape))
    #     model.add(MaxPooling2D(pool_size=(2, 2),strides=(3,2)))
    #     model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',strides=(1,1), padding='same'))
    #     model.add(MaxPooling2D(pool_size=(2, 2),strides=(3,2)))
    #     model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',strides=(1,1), padding='same'))
    #     model.add(Flatten())
    #     model.add(Dense(latent_dim, activation="relu"))
    #     img = Input(shape=img_shape)
    #     validity = model(img)
    #     return Model(img, validity)
    # def build_lstm():
    #     model = Sequential()
    #     model.add(Bidirectional(LSTM(units=120, input_shape=(nb_time_steps, nb_input_vector))))
    #     model.add(Dense(6, activation="softmax"))
    #     encoded_repr = Input(shape=(nb_time_steps, nb_input_vector))
    #     validity = model(encoded_repr)
    #     return Model(encoded_repr, validity)

    opt = Adam(0.0002, 0.5)
    cnn = build_cnn(img_shape)
    rnn = build_rnn()
    dis = build_dis()
    rnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    dis.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    img3 = Input(shape=img_shape)
    encoded_repr3 = cnn(img3)

    def get_class(x):
        return x[:, :, :latent_dim]

    def get_dis(x):
        return x[:, :, latent_dim:]

    encoded_repr3_class = Lambda(get_class)(encoded_repr3)
    encoded_repr3_dis = Lambda(get_dis)(encoded_repr3)
    validity1 = rnn(encoded_repr3_class)
    validity2 = dis(encoded_repr3_dis)
    crnn_model = Model(img3, validity1)
    crnn_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    dis_model = Model(img3, validity2)
    dis_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # crnn_model.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000crnn_model.h5')
    # dis.load_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000dis.h5')
    k = 0
    for epoch in range(epochs):

        idx = np.random.randint(0, train_feature.shape[0], batch_size)
        imgs = train_feature[idx]
        d_loss = dis_model.train_on_batch(imgs, train_domain_label[idx])
        crnn_loss = crnn_model.train_on_batch(imgs, train_label[idx])

        if epoch % 10 == 0:
            print("%d [活动分类loss: %f,acc: %.2f%%,域分类loss: %f,acc: %.2f%%,]" % (
            epoch, crnn_loss[0], 100 * crnn_loss[1], d_loss[0], 100 * d_loss[1]))
            n = 0
            a_all = np.zeros((6, 2))
            for o in range(9):
                print(test_feature.shape)
                non_mid = crnn_model.predict(test_feature[o * 5 * 6:(o + 1) * 5 * 6])
                non_pre = non_mid#(30,2)
                m = 0
                #a = np.zeros((6, 6))
                for i in range(6):
                    for k in range(5):
                        x = np.argmax(non_pre[i * 5 + k])
                        #a[i][x] = a[i][x] + 1
                        a_all[i][x] = a_all[i][x] + 1
                        if ((x == 0 and i<=3)or(x == 1 and i>=3)):
                            m = m + 1
                            n = n + 1
                acc = float(m) / float(len(non_pre))
                print("源" + str(o + 1) + "测试数据准确率：" + str(acc))
                # print(a)
            ac = float(n) / float(270)
            k1 = ac
            print("源平均测试数据准确率：" + str(ac))
            print(a_all)

            non_mid = crnn_model.predict(train_feature_ot)
            non_pre = non_mid
            m = 0
            b = np.zeros((6, 2))
            for i in range(6):
                for k in range(20):
                    x = np.argmax(non_pre[i * 20 + k])
                    b[i][x] = b[i][x] + 1
                    if ((x == 0 and i <= 3) or (x == 1 and i >= 3)):
                        m = m + 1
            acc = float(m) / float(len(non_pre))
            print("目标测试数据准确率：" + str(acc))
            print(b)
            kk1 = acc

            # if ((acc_non_pre3_vot >= 0.66) and (acc_yes_pre3_vot >= 0.66) and (c_loss[1] >= 0.65) and (
            #         acc_non_pre4_vot >= 0.66) and (acc_yes_pre4_vot >= 0.66)):
        #     if ((k1 >= 0.90) and (kk1 >= 0.35) and (d_loss[1] >= 0.90)):
        #         k1 = k1 * 1000
        #         k1 = int(k1)
        #
        #         kk1 = kk1 * 1000
        #         kk1 = int(kk1)
        #         c = 100 * crnn_loss[1]
        #         c = int(c)
        #         d = 100 * d_loss[1]
        #         d = int(d)
        #         file = r'/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/result_crnn-dislstm-to-4.txt'
        #         f = open(file, "ab+")  # 可读可写二进制，文件若不存在就创建
        #         str1 = str(epoch) + 'mid_' + str(c) + 'd' + str(d) + 'y_' + str(
        #             k1) + 'm' + str(kk1) + '\n'
        #         f.write(str1.encode())
        #         f.close()  # 关闭文件
        # if epoch == 1000:
        #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/1000crnn_model.h5')
        #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/1000dis.h5')
        # if epoch == 2000:
        #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/2000crnn_model.h5')
        #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/2000dis.h5')
        # if epoch == 3000:
        #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/3000crnn_model.h5')
        #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/3000dis.h5')
        # if epoch == 4000:
        #     crnn_model.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000crnn_model.h5')
        #     dis.save_weights('/content/drive/MyDrive/huodong/to-1/models/crnn-dislstm-to-4/4000dis.h5')

    localtime2 = time.asctime(time.localtime(time.time()))
    print("开始时间为 :", localtime1)
    print("结束时间为 :", localtime2)

train()
