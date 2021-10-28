import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def dtw(source, target, label, unlabel):
    score = np.zeros([len(target), len(source)])
    # 计算每个样本与模板之间的距离
    for i in range(len(target)):
        for j in range(len(source)):
            distance, _ = fastdtw(target[i, :], source[j, :], dist=euclidean)
            score[i, j] = distance
        print(i)
    # 距离排序，计算相似度得分
    tlabel = np.argmax(label, 1)
    index = np.argmin(score, 1)
    plabel = np.zeros([len(index), 1])
    for i in range(len(index)):
        plabel[i, 0] = tlabel[index[i]]
    acc = plabel - np.argmax(unlabel, 1)
    print("伪标签精度：", np.sum(acc == 0) / len(acc))
    for i in range(len(score)):
        print(unlabel[i], plabel[i])


x = np.genfromtxt('datasets/dtw_test_data/420_dtw_data.csv', dtype=float, delimiter=',',
                         encoding='utf-8')
y = np.genfromtxt('datasets/dtw_test_data/420_dtw_label.csv', dtype=float, delimiter=',',
                          encoding='utf-8')

label_data = np.zeros([64, 200])
unlabel_data = np.zeros([256, 200])
label = np.zeros([64, 4])
unlabel = np.zeros([256, 4])
m,n = 0,0
for i in range(0,320):
    if i % 5 == 0:
        label_data[m, :] = x[i, :]
        label[m, :] = y[i, :]
        m = m + 1
    else:
        unlabel_data[n, :] = x[i, :]
        unlabel[n, :] = y[i, :]
        n = n + 1
print(label_data.shape, label.shape, unlabel_data.shape, unlabel.shape)
dtw(label_data, unlabel_data, label, unlabel)