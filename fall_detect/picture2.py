import numpy as np
import matplotlib.pyplot as plt
import pywt
img_shape=(3,3,30)
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
def dwt(data, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet('sym5')#选取小波函数
    a = data
    ca = []#近似分量
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)#进行5阶离散小波变换
        print(len(a))
        ca.append(a)
    #len(ca)=len(cd)=5,ca[0]
    rec_a = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构
    print("len(rec_a)",len(rec_a))
    for i in range(len(rec_a)):
        print(len(rec_a[i]))
    plt.plot(rec_a[0])
    plt.plot(rec_a[1])
    plt.plot(rec_a[2])
    plt.show()
    return rec_a[4]
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
def data_cut(data,sampleNum=200):
    all_index = np.zeros(data.shape[1])
    var_max_all = 0
    for i in range(data.shape[1]):
        max_var = 0
        max_index = 0
        #print("i",i,"i")
        for j in range(data.shape[0]-sampleNum):
            current_var=np.var(data[j:j+sampleNum,i:i+1])
            #print(current_var)
            if(current_var>=max_var):
                max_var=current_var
                max_index=j
        all_index[i]=max_index
        if (max_var >= var_max_all):
            var_max_all = max_var
    print("all_index",all_index)
    mean_index=int(np.mean(all_index))
    print(mean_index)
    print("var_max_all", var_max_all)
    # var_all=0
    # for i in range(data.shape[1]):
    #     var_all=var_all+np.var(data[mean_index:mean_index+sampleNum,i:i+1])
    # print("var_all",var_all)
    return data[mean_index:mean_index+sampleNum,:]
def data_cut_walk(data,sampleNum=200):

    #num_cut1=int((data.shape[0]-450)/2)
    num_cut1=50
    data=data[num_cut1:num_cut1+400,:]
    idx = np.array([j for j in range(0 ,400, 2)])
    data=data[idx]
    print(data.shape)
    return data
def data_cut2(data,sampleNum=200):
    #计算每个子载波的每个滑动窗口的方差，然后相加比较最大方差
    var_num=np.zeros((data.shape[0]-sampleNum)*data.shape[1])
    var_num=var_num.reshape(data.shape[1],data.shape[0]-sampleNum)
    all_index = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        max_var = 0
        max_index = 0
        for j in range(data.shape[0]-sampleNum):
            current_var=np.var(data[j:j+sampleNum,i:i+1])
            var_num[i:i+1,j:j+1]=current_var
    mean_var_num=np.mean(var_num,axis=1)
    mean_index=int(mean_var_num.argmax())
    print(mean_index)
    return data[mean_index:mean_index+sampleNum,:]
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
def pre_handle1(raw_data):
    data_flag = 0
    for i in range(0, raw_data.shape[1]):
        #print(i)
        data1 = raw_data[:, i].reshape(-1)
        datatmp = smooth(hampel(data1), 25)
        #datatmp = hampel(data1)
        #datatmp = smooth(data1,25)
        if data_flag == 0:
            data = datatmp.reshape(-1, 1)
            data_flag = 1
        else:
            data = np.column_stack((data, datatmp.reshape(-1, 1)))
    print("data_shape:", data.shape)
    print("数据已预处理")
    return data
path1=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125\ly_dd4.dat"
path2=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125\ly_dd5.dat"
path3=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1125\ly_dd6.dat"
path4=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1126_test\zb_dd1.dat"
path5=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1126_test\zb_dd2.dat"
path6=r"D:\my bad\CSI_DATA\fall_detection\fall_detection\data_model_dir\data_dir\1126_test\zb_dd3.dat"
data1=get_data(path1)
data2=get_data(path2)
data3=get_data(path3)
data4=get_data(path4)
data5=get_data(path5)
data6=get_data(path6)

data7=np.concatenate((data1,data2,data3,data4,data5,data6),axis=0)

#sss=dwt(data1[:, 10:11],data1.shape[0])
#hhh=data1[:, 10:11]
data1=pre_handle(data1)
data2=pre_handle(data2)
data3=pre_handle(data3)
data4=pre_handle(data4)
data5=pre_handle(data5)
data6=pre_handle(data6)
#data7=pre_handle(data7)



# data1=data_cut(data1)
# data2=data_cut(data2)
# data3=data_cut(data3)
# data4=data_cut_walk(data4)
# data5=data_cut_walk(data5)
# data6=data_cut_walk(data6)

data1=np.array(data1, dtype=np.float64)
data2=np.array(data2, dtype=np.float64)
data3=np.array(data3, dtype=np.float64)
data4=np.array(data4, dtype=np.float64)
data5=np.array(data5, dtype=np.float64)
data6=np.array(data6, dtype=np.float64)
data7=np.array(data7, dtype=np.float64)
print(data1.shape)
# print(data2.shape)
# print(data3.shape)
# plt.plot(data1)
# plt.show()
# plt.plot(data2)
# plt.show()
# plt.plot(data3)
# plt.show()
# plt.plot(data4)
# plt.show()
# plt.plot(data5)
# plt.show()
# plt.plot(data6)
# plt.show()

data1 = data1[:, 10:11]
data2 = data2[:, 10:11]
data3 = data3[:, 10:11]
data4 = data4[:, 10:11]
data5 = data5[:, 10:11]
data6 = data6[:, 10:11]
print(data1.shape)


#data7 = data7[:, 10:11]
t = range(10000)
plt.plot(t[:data1.shape[0]], data1, 'r')
plt.plot(t[:data2.shape[0]], data2, 'g')
plt.plot(t[:data3.shape[0]], data3, 'b')
plt.plot(t[:data4.shape[0]], data4, 'y')
plt.plot(t[:data5.shape[0]], data5, 'black')
plt.plot(t[:data6.shape[0]], data6, 'pink')
#plt.plot(t[:data7.shape[0]], data7, 'm')
plt.show()