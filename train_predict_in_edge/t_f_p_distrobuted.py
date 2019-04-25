
import pandas as pd
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
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



# 划分训练集、测试集
x_train_1, x_train, y_train_1, y_train = train_test_split(x_train, y_train, test_size=0.25)
x_train_2, x_train, y_train_2, y_train = train_test_split(x_train, y_train, test_size=1/3)
x_train_3, x_train_4, y_train_3, y_train_4 = train_test_split(x_train, y_train, test_size=0.5)

# 在云端进行数据训练
'''
# 1.用哪种模型，采用何种调参方法
'''

start_time = time.time()
# 模型1
lin_m_1 = LinearRegression()
# lin_m_1 = Lasso()
lin_m_1.fit(x_train_1, y_train_1)

# 模型2
lin_m_2 = LinearRegression()
# lin_m_2 = Lasso()
lin_m_2.fit(x_train_2, y_train_2)

# 模型3
lin_m_3 = LinearRegression()
# lin_m_3 = Lasso()
lin_m_3.fit(x_train_3, y_train_3)

# 模型4
lin_m_4 = LinearRegression()
# lin_m_4 = Lasso()
lin_m_4.fit(x_train_4, y_train_4)


m_inter = []
m_ceof = []
# print(lin_m_1.intercept_)
# print(lin_m_1.coef_)
# 数据聚合
m_i = [lin_m_1, lin_m_2, lin_m_3, lin_m_4]
for i in range(4):
    m_inter.append(m_i[i].intercept_)
    m_ceof.append(m_i[i].coef_)
# print(m_inter)
# print(m_ceof )
# 求均值
end_m_inter = np.mean(m_inter, axis=0)
end_m_ceof =  np.mean(m_ceof, axis=0)

# 预测
def predict_x(x_test_1, m_int, m_ce):
    ONE = np.array(m_ce, ndmin=2)
    two = np.array(x_test_1).T
    return (np.reshape(m_int + np.dot(ONE, two), (-1, 1)))[0, 0]




y_pre = []
# 0
y_pre.append(predict_x(x_test[0], end_m_inter, end_m_ceof))

# 1
x_test_1= x_test[1]
x_test_1[2] = y_pre[0]

y_pre.append(predict_x(x_test_1, end_m_inter, end_m_ceof))
# 2
x_test_2 = x_test[2]
x_test_2[2] = y_pre[1]
x_test_2[3] = y_pre[0]
y_pre.append(predict_x(x_test_2, end_m_inter, end_m_ceof))


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
    y_pre.append(predict_x(x_test_i, end_m_inter, end_m_ceof))

endtime = time.time()
process_time = (endtime - start_time) * 1000
print(y_pre)
print('mse:', mean_squared_error(y_test, y_pre))
print('rmse:', np.sqrt(mean_squared_error(y_test, y_pre)))
print('耗时：', process_time)


plt.figure(figsize=(16, 9))
# 假设网络带宽：4800kb/s， 传输距离2000KM， 电信号传播速度光速的2/3   = 2e+5km/s
# 计算训练集、测试集数据的大小
size_train_data = sys.getsizeof(x_train) + sys.getsizeof(y_train)
size_test_data = sys.getsizeof(x_test)

plt.plot(y_test[:576], '.', c=(0, 0, 1), label='%s' % str('true'))
plt.plot(y_pre[:576], lw=1.0, label='%s' % str('pre'))

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