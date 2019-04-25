
import pandas as pd
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from func_model.BO_model import opt
import func_model.light_GBM as lgb
from func_model.diatance_trance import time_FS, time_TB
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv('../data/traffic_flow_p/data_train.csv')
data_test = pd.read_csv('../data/traffic_flow_p/data_test.csv')

# ['NSM', 'week', 'd_t_1', 'd_t_2', 'd_t_3', 'Veh']

x_train = data_train[['NSM', 'week', 'd_t_1', 'd_t_2', 'd_t_3']].values
y_train = np.reshape(data_train[['Veh']].values, (1, -1))[0]


m, n = np.shape(data_test[['d_t_1', 'd_t_2', 'd_t_3']].values)

zero_list = list(np.zeros(m))
for i in ['d_t_1', 'd_t_2', 'd_t_3']:

    data_test[i] = zero_list
x_test = data_test[['NSM', 'week', 'd_t_1', 'd_t_2', 'd_t_3']].values


y_test = np.reshape(data_test[['Veh']].values, (1, -1))[0]


# 在云端进行数据训练
'''
# 1.用哪种模型，采用何种调参方法
'''
start_time = time.time()
opt_x, opt_y = opt(lgb, max_iters=30, train_x=x_train, train_y=y_train, cv_mun=10, init_start=5, model_score='neg_mean_squared_error', print_opt=True)
lgb_f = lgb.f(opt_x)
lgb_f.fit(x_train, y_train)

y_pre = []
# 0
y_pre.append(lgb_f.predict(np.array(x_test[0], ndmin=2)))

# 1

x_test_1 = x_test[1]
x_test_1[2] = y_pre[0]
x_test_1[3] = 0
x_test_1[4] = 0


y_pre.append(lgb_f.predict(np.array(x_test_1, ndmin=2)))
# 2
x_test_2 = x_test[2]
x_test_2[2] = y_pre[1]
x_test_2[3] = y_pre[0]
x_test_2[4] = 0
y_pre.append(lgb_f.predict(np.array(x_test_2, ndmin=2)))


d_NSM = data_test['NSM'].values
d_week = data_test['week'].values
for i in range(3, m):
    x_test_i = x_test[i]
    if d_NSM[i] - d_NSM[i-1] == 1 or abs(d_week[i] - d_week[i-1]) == 1:
        x_test_i[2] = y_pre[i-1]
    else:
        x_test_i[2] = 0
    if d_NSM[i] - d_NSM[i-2] == 2 or abs(d_week[i] - d_week[i-2]) == 1:
        x_test_i[3] = y_pre[i-2]
    else:
        x_test_i[3] = 0
    if d_NSM[i] - d_NSM[i-3] == 3 or abs(d_week[i] - d_week[i-3]) == 1:
        x_test_i[4] = y_pre[i-3]
    else:
        x_test_i[4] = 0
    # print(x_test_i)
    y_pre.append(lgb_f.predict(np.array(x_test_i, ndmin=2)))

endtime = time.time()
process_time = (endtime - start_time) * 1000

print('mse:', mean_squared_error(y_test, y_pre))
print('rmse:', np.sqrt(mean_squared_error(y_test, y_pre)))
print('耗时：', process_time)


plt.figure(figsize=(16, 9))
# 假设网络带宽：4800kb/s， 传输距离2000KM， 电信号传播速度光速的2/3   = 2e+5km/s
# 计算训练集、测试集数据的大小
size_train_data = sys.getsizeof(x_train) + sys.getsizeof(y_train)
size_test_data = sys.getsizeof(x_test)

plt.plot(y_test[:576], '.', c=(0, 0, 1), label='%s' % str('true'))
plt.plot(y_pre[:576], lw=1.5, color='black', label='%s' % str('pre'))

plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('Veh')
plt.xlabel('time')
plt.title('traffic flow prediction')

plt.show()
# print('size_train_data:%s bit' %size_train_data, end=' ')
# print('size_test_data:%s bit' %size_test_data)
# print('训练数据的延迟：', time_FS(data_len=size_train_data, speed_fs=4800e+3)*1000 + time_TB(dis_point=2e+3, speed_tb=2e+5)*1000)
# print('测试数据的延迟：', time_FS(data_len=size_test_data, speed_fs=4800e+3)*1000 + time_TB(dis_point=2e+3, speed_tb=2e+5)*1000)







