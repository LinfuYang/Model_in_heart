from the_best_al_in_world.AGPR import A_GPR
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
import time
import matplotlib.pyplot as plt
from func_model.diatance_trance import time_FS, time_TB
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv('../data/traffic_flow_p/data_train.csv')
data_test = pd.read_csv('../data/traffic_flow_p/data_test.csv')

# ['NSM', 'week', 'd_t_1', 'd_t_2', 'd_t_3', 'Veh']

f_s = StandardScaler().fit(X=(data_train['NSM'].values).reshape(-1, 1))

# data_train['NSM'] = f_s.transform((data_train['NSM'].values).reshape(-1, 1))
x_train = data_train[['NSM', 'week', 'd_t_1', 'd_t_2', 'd_t_3']].values
y_train = np.reshape(data_train[['Veh']].values, (1, -1))[0]

# data_test['NSM'] = f_s.transform((data_test['NSM'].values).reshape(-1, 1))
x_test = data_test[['NSM', 'week', 'd_t_1', 'd_t_2', 'd_t_3']].values
y_test = np.reshape(data_test[['Veh']].values, (1, -1))[0]




m, n = np.shape(x_train)
print('训练数据总量', m)

_, x_train_l, _, y_train_l = train_test_split(x_train, y_train, test_size=50/m)
print('低似真度数据样本量：', np.shape(x_train_l)[0])

# 为 low fidelity 数据添加噪声
mu = 0
sigma = 10
for i in range(np.shape(x_train_l)[0]):
    # y_train_l[i] = y_train_l[i] + random.gauss(0, sigma)
    y_train_l[i] = y_train_l[i] * 1.2 - 10

x_train_l = np.array(x_train_l, ndmin=2)
y_train_l = np.reshape(y_train_l, (-1, 1))


a_gpr = A_GPR()
init_size = 2
x_init_h, y_init_h = a_gpr.sample_point(x_data_conda=x_train, y_data_conda=np.array(y_train).reshape(-1, 1), iter=init_size, is_init=True)
max_iter = 300

left = 300
right =301
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

                                               )
for j in range(len(list_arr)):
    list_mse_gpr.append(np.sqrt(mean_squared_error(y_test, y_pre[j])))
list_w.append(list_w_hf)
print(list_w)
end = time.time()
# print(list_w)
print(list_mse_gpr)
# print((end - start) * 1000)
plt.figure(figsize=(16, 9))
# 假设网络带宽：4800kb/s， 传输距离2000KM， 电信号传播速度光速的2/3   = 2e+5km/s
# 计算训练集、测试集数据的大小
size_train_data = sys.getsizeof(x_train) + sys.getsizeof(y_train)
size_test_data = sys.getsizeof(x_test)

plt.plot(y_test[:576], '.', c=(0, 0, 1), label='%s' % str('true'))
plt.plot((y_pre[0])[:576], lw=1.5, color='black', label='%s' % str('pre'))

plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('Veh')
plt.xlabel('time')
plt.title('traffic flow prediction')

plt.show()
# 假设网络带宽：4800b/s， 传输距离2000KM， 电信号传播速度光速的2/3   = 20Wkm/s

# 计算训练集、测试集数据的大小
size_train_data = sys.getsizeof(x_train) + sys.getsizeof(y_train)
size_test_data = sys.getsizeof(x_test) + sys.getsizeof(y_test)
# print('发送延迟：', time_FS(data_len=size_train_data, speed_fs=4800e+3)*1000)
# print('传播延迟：', time_TB(dis_point=2e+3, speed_tb=2e+5)*1000)