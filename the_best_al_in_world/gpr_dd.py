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

data_train = pd.read_csv('../data/training.csv')
data_test = pd.read_csv('../data/testing_validation.csv')
column_data = list(data_train.columns)

# 数据处理，类别数据

data_train['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_train['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

data_test['WeekStatus'].replace(['Weekday', 'Weekend'], [1, 2], inplace=True)
data_test['Day_of_week'].replace(['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
                                  , [1, 2, 3, 4, 5, 6, 7], inplace=True)

column = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
             'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',  'Tdewpoint', 'NSM',  'Day_of_week']

column_1 = ['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
            'T_out','Press_mm_hg', 'RH_out', 'Windspeed', 'Tdewpoint', 'NSM', 'Day_of_week'
            ]
std = ['NSM']

for index in std:
    scaler = StandardScaler()
    data_train[index] = scaler.fit_transform((data_train[index].values).reshape(-1, 1))
for index in std:
    scaler = StandardScaler()
    data_test[index] = scaler.fit_transform((data_test[index].values).reshape(-1, 1))



# 划分训练集、测试集
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

x_train = data_train[column_1].values
x_test  = data_test[column_1].values

y_train = data_train['Appliances'].values
y_test = data_test['Appliances'].values


m, n = np.shape(x_train)
print('训练数据总量', m)

_, x_train_l, _, y_train_l = train_test_split(x_train, y_train, test_size=200/m)
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
                                               n_start=2,
                                               n_single=100
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
