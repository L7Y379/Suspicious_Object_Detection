# # univariate cnn-lstm example
# import numpy as np
# from numpy import array
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import TimeDistributed
# from keras.layers.convolutional import Conv1D,Conv2D,MaxPooling2D
# from keras.layers.convolutional import MaxPooling1D
# # define dataset
# X = np.arange(1600)
# y = array([50, 60, 70, 80])
# # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
# X = X.reshape((4, 10, 5,8, 1))
# # define model
# model = Sequential()
# model.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3), activation='relu'), input_shape=(10, 5,8, 1)))
# model.add(TimeDistributed(MaxPooling2D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# model.add(LSTM(50, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# # fit model
# model.fit(X, y, epochs=500, verbose=0)
# # demonstrate prediction
# x_input = np.arange(1600)
# x_input = x_input.reshape((4, 10, 5,8, 1))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)

import numpy as np
import pandas as pd
# a=np.arange(10)
# b=np.arange(10)
# c=np.arange(10)
# a[0]=95.1
# a[1]=95.1
# a[2]=93.7
# a[3]=94.8
# a[4]=95.5
# a[5]=92.5
# a[6]=90.3
# a[7]=94
# a[8]=95.1
# a[9]=94
# print(np.mean(a))
#
# b[0]=99
# b[1]=98
# b[2]=97
# b[3]=98
# b[4]=99
# b[5]=99
# b[6]=97
# b[7]=99
# b[8]=95
# b[9]=100
# print(np.mean(b))
#
# c[0]=75.8
# c[1]=65
# c[2]=66.6
# c[3]=33.3
# c[4]=65
# c[5]=33.3
# c[6]=50.8
# c[7]=75
# c[8]=58.3
# c[9]=64.1
# print(np.mean(c))
fn = 'D:/my bad/CSI_DATA/cy/Matlab/empty_env_csv/empty_csi_5f.csv'
csvdata = pd.read_csv(fn, header=None)
csvdata = np.array(csvdata, dtype=np.float64)
print(csvdata.shape)

fn = 'E:\cy\Matlab\empty_env_csi/empty_csi_5f.csv'
csvdata = pd.read_csv(fn, header=None)
csvdata = np.array(csvdata, dtype=np.float64)
print(csvdata.shape)
