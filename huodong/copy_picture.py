import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data1 = pd.read_csv('D:\\my bad\\CSI_DATA\\test_tk_path_1027\\smooth_data\\smooth_data\\two_open_100_1.csv',error_bad_lines=False)#red
data1= np.array(data1, dtype=np.float64)
data1 = data1[1000:2000, 50:51]
t = range(10000)
plt.plot(t[:data1.shape[0]], data1, 'r')
plt.show()