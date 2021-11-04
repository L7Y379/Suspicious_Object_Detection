
import numpy as np
import pandas as pd
fn1 = 'D:/my bad/Suspicious object detection/data/huodong/room1/room1_phase_data.csv'
csvdata = pd.read_csv(fn1, header=None)
csvdata = np.array(csvdata, dtype=np.float64)
print(csvdata.shape)#(1200,18000)

