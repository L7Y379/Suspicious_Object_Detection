#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

hhh=np.arange(100)
hhh=hhh.reshape(20,5)


ecg = pywt.data.ecg()

data1 = np.concatenate((np.arange(1, 400),
                        np.arange(398, 600),
                        np.arange(601, 1024)))
x = np.linspace(0.082, 2.128, num=1024)[::-1]
data2 = np.sin(40 * np.log(x)) * np.sign((np.log(x)))
print(ecg.shape)
print(data1.shape)
print(data2.shape)
mode = pywt.Modes.smooth


def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)#选取小波函数
    a = data
    ca = []#近似分量
    cd = []#细节分量
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)#进行5阶离散小波变换
        print(len(a))
        print(len(d))
        ca.append(a)
        cd.append(d)
    #len(ca)=len(cd)=5,ca[0]

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))

    # print("len(rec_a)",len(rec_a))
    # print("len(rec_d)",len(rec_d))
    # for i in range(len(rec_a)):
    #     print(len(rec_a[i]))
    # for i in range(len(rec_d)):
    #     print(len(rec_d[i]))
    #
    # plt.plot(rec_a[4])
    # plt.show()

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))


# plot_signal_decomp(data1, 'coif5', "DWT: Signal irregularity")
# plot_signal_decomp(data2, 'sym5',
#                   "DWT: Frequency and phase change - Symmlets5")
plot_signal_decomp(ecg, 'sym5', "DWT: Ecg sample - Symmlets5")


plt.show()


path1=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125\tk_dd1.dat"
path2=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125\tk_dd2.dat"
path3=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125\tk_dd3.dat"
path4=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125_test\tk_dd4.dat"
path5=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125_test\tk_dd5.dat"
path6=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125_test\tk_dd6.dat"
img_shape=(3,3,30)
def get_data(path):
    curpin = 0
    stream, curpin = read_bf_file(path, curpin) ##stream里包含的是幅值了么  curpin是啥
    amplitude_list = []
    for i in range(len(stream)):
        data = stream[i]["csi"]
        if data.shape == img_shape:
            amplitude_list.append(abs(data))
    print("amplitude_list:",len(amplitude_list))
    amplitude_list = np.array(amplitude_list)
    stream_len = len(amplitude_list)
    amplitude_list_new = amplitude_list.reshape((stream_len,-1))
    #amplitude_list_new=data_cut(amplitude_list_new,sampleNum=200)
    #amplitude_list_new = amplitude_list_new[200:200+sampleNum] ##获取1000条数据是这样么
    #
    return amplitude_list_new
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
def read_bf_file(filename, curpin, decoder="python"):
    with open(filename, "rb") as f:
        # 读取文件大小
        f.seek(0, 2)
        size = f.tell()
        # print("我们文件的大小是--"+str(size))
        f.seek(curpin, 0)
        bfee_list = []
        field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
        while field_len != 0 and size - f.tell() > 572:
            bfee_list.append(f.read(field_len))
            field_len = int.from_bytes(f.read(2), byteorder='big', signed=False)
        if f.tell() == size:
            if field_len == 0:
                curpin = f.tell()
            else:
                curpin = f.tell() - 2
        else:
            curpin = f.tell() - 2
    dicts = []

    count = 0  # % Number of records output
    broken_perm = 0  # % Flag marking whether we've encountered a broken CSI yet
    triangle = [0, 1, 3]  # % What perm should sum to for 1,2,3 antennas

    csi_len = len(bfee_list)
    for array in bfee_list[:csi_len]:
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

    return dicts, curpin
def pre_handle(raw_data):
    data_flag = 0
    for i in range(0, raw_data.shape[1]):
        #print(i)
        data1 = raw_data[:, i].reshape(-1)
        #datatmp = smooth(hampel(data1), 25)
        #datatmp = hampel(data1)
        datatmp = smooth(data1,25)
        if data_flag == 0:
            data = datatmp.reshape(-1, 1)
            data_flag = 1
        else:
            data = np.column_stack((data, datatmp.reshape(-1, 1)))
    print("data_shape:", data.shape)
    print("数据已预处理")
    return data
def smooth(a, WSZ):
    # a:原始数据，NumPy 1-D array containing the data to be smoothed
    # 必须是1-D的，如果不是，请使用 np.ravel()或者np.squeeze()转化
    # WSZ: smoothing window size needs, which must be odd number,窗口尺寸必须是奇数
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))
data1=get_data(path1)
# data2=get_data(path2)
# data3=get_data(path3)
# data4=get_data(path4)
# data5=get_data(path5)
# data6=get_data(path6)
data1=pre_handle(data1)
plot_signal_decomp(data1[:, 10:11], 'sym5', "DWT: Ecg sample - Symmlets5")
plt.show()