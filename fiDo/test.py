import numpy as np
l = [3, 2, 1, 0, 4, 5]

idx1 = np.array([j for j in range(0 ,3, 1)])
idx2=np.array([j for j in range(4 ,5, 1)])
idx=np.hstack((idx1,idx2))
print(idx)
l.index(max(l))  # 返回list最大值位置
#idx = np.array([4,2])


array_l = np.array(l)
print(array_l[idx])
a=np.argmax(array_l)  # 返回array最大值位置
idx=np.array([j for j in range(0 ,a, 1)])
print(array_l[idx,])
print(a)