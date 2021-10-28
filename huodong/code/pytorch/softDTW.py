import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def SDTW(s, t, sl, tl, c):
    # 划分初始类中心
    center = {} # 类中心模板
    for i in range(len(sl)):
        if sl[i] in center:
            center[sl[i]].append(i)
        else:
            center[sl[i]] = [i]

    # 初始DTW距离计算
    pLabel = np.zeros([len(tl), c]) # 保存贴好的伪标签距离
    pLabel2 = np.zeros([len(tl), c])
    noLabel = set(range(len(tl)))  # 无标签集序号
    for i in noLabel:
        for j in range(c):
            for k in center[j]:
                d,_ = fastdtw(t[i], s[k], dist=euclidean)
                if pLabel[i, j] == 0 or d > pLabel[i, j]:
                    pLabel[i, j] = d
                    pLabel2[i, j] = d

    # 迭代式DTW距离计算
    while len(noLabel) > 0:
        tmp_lab = np.argmin(pLabel2, 0)
        for i in range(c):
            center[i] = [tmp_lab[i]]
            pLabel2[tmp_lab[i]] = 100000
            if tmp_lab[i] in noLabel:
                noLabel.remove(tmp_lab[i])
        print(center)
        for i in noLabel:
            for j in range(c):
                for k in center[j]:
                    d,_ = fastdtw(t[i], t[k], dist=euclidean)
                    if pLabel[i, j] == 0 or d > pLabel[i, j]:
                        pLabel[i, j] = d
                        pLabel2[i, j] = d
    return pLabel

x = np.genfromtxt('E:/CSI自采数据集/手势数据集/重采样后数据集/room2人变化/data.csv', dtype=float, delimiter=',',encoding='utf-8')
y = np.genfromtxt('E:/CSI自采数据集/手势数据集/重采样后数据集/room2人变化/label.csv', dtype=float, delimiter=',',encoding='utf-8')
idx =600
label_data = x[idx:idx+120:4,:108000:540]
label = y[idx:idx+120:4]
unlabel_data = np.concatenate([x[idx+1:idx+120:4,:108000:540], x[idx+2:idx+120:4,:108000:540], x[idx+3:idx+120:4,:108000:540]])
unlabel = np.concatenate([y[idx+1:idx+120:4], y[idx+2:idx+120:4], y[idx+3:idx+120:4]])
# label_data = np.zeros([32, 200])
# unlabel_data = np.zeros([288, 200])
# label = np.zeros([32, 4])
# unlabel = np.zeros([288, 4])
# m,n = 0,0
# for i in range(0,320):
#     if i % 10 == 0:
#         label_data[m, :] = x[i, :]
#         label[m, :] = y[i, :]
#         m = m + 1
#     else:
#         unlabel_data[n, :] = x[i, :]
#         unlabel[n, :] = y[i, :]
#         n = n + 1

print(label_data.shape, label.shape, unlabel_data.shape, unlabel.shape)
pL = SDTW(label_data, unlabel_data, np.argmax(label,1), np.argmax(unlabel,1), 6)
pS = np.zeros_like(pL)
for i in range(len(pL)):
    pr = 1 / pL[i, :]
    pS[i, :] = pr / np.sum(pr)
    # print('距离：', pL[i,:], '\t概率：', pS[i, :], '\t伪标签：', np.argmin(pL[i,:]), '真实标签：', np.argmax(unlabel[i, :]))
# np.savetxt('E:/gzyPytorchWork/result/420_dtw_hard_label1.csv', pS, fmt='%.3f', delimiter=',', encoding='utf-8')
np.savetxt('E:/gzyPytorchWork/result/plabel6.csv', pS, fmt='%.3f', delimiter=',', encoding='utf-8')
acc = np.argmin(pL, 1)-np.argmax(unlabel, 1)
print(len(acc[acc==0])/len(unlabel))


