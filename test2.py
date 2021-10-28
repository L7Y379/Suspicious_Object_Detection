
import numpy as np
temp_label = np.tile(0, (60,))
temp_label2 = np.tile(1, (30,))
temp_label=np.concatenate((temp_label, temp_label2), axis=0)
for i in range(8):
    temp_label = np.concatenate((temp_label, temp_label), axis=0)
print(temp_label.shape)
print(temp_label)

