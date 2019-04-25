from the_best_al_in_world.AGPR_1 import A_GPR_1
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
import time
from func_model.diatance_trance import time_FS, time_TB
import warnings
warnings.filterwarnings('ignore')



data = pd.read_excel('../data/cccp_data.xlsx')


column = ['AT', 'V', 'AP', 'RH']
for index in column:
    scaler = MinMaxScaler()
    data[index] = scaler.fit_transform((data[index].values).reshape(-1, 1))

# ['AT', 'V', 'AP', 'RH', 'PE']
x_data = data[['AT', 'V', 'AP', 'RH']].values
y_data = np.reshape(data[['PE']].values, (1, -1))[0]

print(data.head(10))

# 划分训练集、测试集
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

m, n = np.shape(x_train)
print('训练数据总量', m)

_, x_train_l, _, y_train_l = train_test_split(x_train, y_train, test_size=60/m)
print('低似真度数据样本量：', np.shape(x_train_l)[0])

# 为 low fidelity 数据添加噪声
mu = 0
sigma = 10
for i in range(np.shape(x_train_l)[0]):
    y_train_l[i] = y_train_l[i] + random.gauss(0, sigma)
    # y_train_l[i] = y_train_l[i] * 1.1 - 10

x_train_l = np.array(x_train_l, ndmin=2)
y_train_l = np.reshape(y_train_l, (-1, 1))


a_gpr = A_GPR_1()
init_size = 2
x_init_h, y_init_h = a_gpr.sample_point(x_data_conda=x_train, y_data_conda=np.array(y_train).reshape(-1, 1), iter=init_size, is_init=True)
max_iter = 150

left = 100
right =151
dist = 50
list_arr = list(range(left, right, dist))


list_w = []
list_mse_gpr = []
start = time.time()
hf_gp, list_w_hf, y_pre = a_gpr.creat_gp_model(max_loop=max_iter, x_init_l=x_train_l, y_init_l=y_train_l,
                                               x_init_h=x_init_h, y_init_h=y_init_h,
                                               x_conda=np.array(x_train, ndmin=2),
                                               y_conda=np.array(y_train).reshape(-1, 1),
                                               list_iter=list_arr, data_test=x_test,
                                               n_start=1,
                                               n_single=200
                                               )
for j in range(len(list_arr)):
    list_mse_gpr.append(mean_squared_error(y_test, y_pre[j]))
list_w.append(list_w_hf)
end = time.time()
print(list_w)
print(list_mse_gpr)
print((end - start) * 1000)

# 假设网络带宽：4800b/s， 传输距离2000KM， 电信号传播速度光速的2/3   = 20Wkm/s

# 计算训练集、测试集数据的大小
size_train_data = sys.getsizeof(x_train) + sys.getsizeof(y_train)
size_test_data = sys.getsizeof(x_test) + sys.getsizeof(y_test)

print('发送延迟：', time_FS(data_len=size_train_data, speed_fs=4800e+3)*1000)
print('传播延迟：', time_TB(dis_point=2e+3, speed_tb=2e+5)*1000)