import math
import os
import pandas as pd

import numpy as np


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

    csi_pwr[csi_pwr == 0] = 1
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
def get_data(path):
    curpin = 0
    stream, curpin = read_bf_file(path, curpin) ##stream里包含的是幅值了么  curpin是啥
    amplitude_list = []
    for i in range(len(stream)):
        data = stream[i]["csi"]
        if data.shape == (3,3,30):
            amplitude_list.append(abs(data))
    print("amplitude_list:",len(amplitude_list))
    amplitude_list = np.array(amplitude_list)
    stream_len = len(amplitude_list)
    amplitude_list_new = amplitude_list.reshape((stream_len,-1))
    #amplitude_list_new = amplitude_list_new[200:200+sampleNum] ##获取1000条数据是这样么

    #
    return amplitude_list_new
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
# hampel滤波器，去除离散的点
def hampel(X):
    length = X.shape[0] - 1
    k = 19
    # k为窗口大小
    nsigma = 3
    iLo = np.array([i - k for i in range(0, length + 1)])
    iHi = np.array([i + k for i in range(0, length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > length] = length
    xmad = []
    xmedian = []
    for i in range(length + 1):
        w = X[iLo[i]:iHi[i] + 1]
        medj = np.median(w)
        mad = np.median(np.abs(w - medj))
        xmad.append(mad)
        xmedian.append(medj)
    xmad = np.array(xmad)
    xmedian = np.array(xmedian)
    scale = 1.4826  # 缩放
    xsigma = scale * xmad
    xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）

    # 将离群点替换为中位数值
    xf = X.copy()
    xf[xi] = xmedian[xi]
    return xf
def pre_handle(raw_data):
    data_flag = 0
    for i in range(0, raw_data.shape[1]):
        #print(i)
        data1 = raw_data[:, i].reshape(-1)
        datatmp = smooth(hampel(data1), 25)

        if data_flag == 0:
            data = datatmp.reshape(-1, 1)
            data_flag = 1
        else:
            data = np.column_stack((data, datatmp.reshape(-1, 1)))
    print("data_shape:", data.shape)
    print("数据已预处理")
    return data
def pre_handle_onlysmooth(raw_data):
    data_flag = 0
    for i in range(0, raw_data.shape[1]):
        #print(i)
        data1 = raw_data[:, i].reshape(-1)
        datatmp = smooth(data1, 25)

        if data_flag == 0:
            data = datatmp.reshape(-1, 1)
            data_flag = 1
        else:
            data = np.column_stack((data, datatmp.reshape(-1, 1)))
    print("data_shape:", data.shape)
    print("数据已预处理")
    return data
def data_cut(data,sampleNum=200):
    all_index = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        max_var = 0
        max_index = 0
        for j in range(data.shape[0]-sampleNum):
            current_var=np.var(data[j:j+sampleNum,i:i+1])
            if(current_var>=max_var):
                max_var=current_var
                max_index=j
        #print(max_var)
        all_index[i]=max_index
    #print("all_index",all_index)
    mean_index=int(np.mean(all_index))
    print("mean_index",mean_index)
    return data[mean_index:mean_index+sampleNum,:]
def pre_datalist(dataList,dirPath,dirPath_pre):
    for i in range(0, len(dataList)):
        path = os.path.join(dirPath, dataList[i])
        if os.path.isfile(path):
            temp_data = get_data(path)
            # 数据预处理
            temp_data = pre_handle_onlysmooth(temp_data)
            # 数据截取
            temp_data = data_cut(temp_data)
            print(temp_data.shape)
            path = os.path.join(dirPath_pre, dataList[i])
            path = path.replace(".dat", ".csv")
            df = pd.DataFrame(temp_data)
            # 保存 dataframe
            df.to_csv(path, encoding='utf-8', index=False, header=None)
            # path=path.replace(".dat",".csv")
            # os.makedirs(path)
            # with open(path, 'w',encoding='utf-8') as f:
            #     f_csv = csv.writer(f)
            #     f_csv.writerows(temp_data)
def pre_data():
    print("pre_data已调用")
    dirPath = "D:/my bad/Suspicious object detection/data/fall/1112_test"
    dirPath_pre="D:/my bad/Suspicious object detection/data/fall/1112_pre_test"
    if not os.path.exists(dirPath_pre):
        os.makedirs(dirPath_pre)
    dataList = os.listdir(dirPath)

    pre_datalist(dataList,dirPath,dirPath_pre)

pre_data()