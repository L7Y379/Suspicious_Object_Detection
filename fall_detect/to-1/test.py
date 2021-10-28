import numpy as np
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=[0,1])


a=np.arange(50)
a=a.reshape(10,5)
print(a)
a= min_max_scaler.fit_transform(a)
print(a)