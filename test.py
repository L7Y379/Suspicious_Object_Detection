from sklearn import preprocessing
import numpy as np
x = np.array([[4.5,-1.,2.],
              [10.,0.,0.],
              [0.,1.,-1.]])
min_max_scaler = preprocessing.MinMaxScaler()#默认为范围0~1，拷贝操作  # 其中默认的参数是 copy = True ，也就是生成新的数据，不改变原来的数据
# min_max_scaler1 = preprocessing.MinMaxScaler(feature_range = (1,3),copy = False)#范围改为1~3，对原数组操作
# min_max_scaler1 = preprocessing.MinMaxScaler(feature_range = (0,1),copy = False)#范围改为1~3，对原数组操作
# min_max_scaler1 = preprocessing.MinMaxScaler(feature_range = (0,1),copy = 0)#范围改为1~3，对原数组操作  # copy = 0 时，完全和 copy=False 时 得到的结果一样
tk_feature=((x.astype('float32')-np.min(x))-(np.max(x)-np.min(x))/2.0)/((np.max(x)-np.min(x))/2)
print(tk_feature)
min_max_scaler1 = preprocessing.MinMaxScaler(feature_range = (-1,1),copy = 1)#  copy =1 时 和 copy = True 时得到的结果一样， 其中copy=True 是系统默认的
x_minmax = min_max_scaler1.fit_transform(x)
print('x_minmax = \n',x_minmax)

print('x = \n', x)

print("--------- 我是分割线   ------------")
#  原来的 x 和 现在的 min_max_scaler_copy 数值是一样的，都是fit_transform 之后的数据
min_max_scaler_copy = min_max_scaler1.fit_transform(x)
print("x = \n", x)
print("min_max_scaler_copy = \n", min_max_scaler_copy)


print("-------- wanjie  -----------")
#新的测试数据进来，同样的转换      # 新的数据，列数一样，行数可以不一样
x_test = np.array([[-3,-1,4.],
                   [0,-1,10]])
x_test_maxabs = min_max_scaler.transform(x_test)
print('x_test_maxabs = ',x_test_maxabs)
