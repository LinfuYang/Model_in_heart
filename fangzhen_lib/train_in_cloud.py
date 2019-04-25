import pandas as pd
import numpy as np
import sys
import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from func_model.BO_model import opt
import func_model.light_GBM as lgb
import func_model.GBDT as gbdt
from func_model.diatance_trance import time_FS, time_TB

import warnings
warnings.filterwarnings('ignore')




from data.fangzhen_data.func_ND import func_4D


def sample_point(round_xy=None, iter=None, sample_tpye='uniform'):
    m, n = np.shape(round_xy)
    x_temp = np.zeros((iter, m))
    if sample_tpye == 'uniform':
        for k in range(iter):
            for i in range(m):
                x_temp[k, i] = np.random.uniform(round_xy[i, 0] + 10e-99, round_xy[i, 1])

    elif sample_tpye == 'linspace':
        for i in range(m):
            x_temp[:, i] = np.linspace(round_xy[i, 0] + 10e-99, round_xy[i, 1], iter)

    return x_temp
f_4d = func_4D()
x_rpund = f_4d.round_x

num_low_data = 10
x_train_l = sample_point(round_xy=x_rpund, iter=num_low_data)
y_train_l = np.reshape([f_4d.f_l(x_train_l[i]) for i in range(num_low_data)], (-1,1))

num_high_data = 500
x_data = sample_point(round_xy=x_rpund, iter=num_high_data)
y_data = [f_4d.f_obj(x_data[i]) for i in range(num_high_data)]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)





# 在云端进行数据训练
'''
1.用哪种模型，采用何种调参方法
'''

start_time = time.time()

opt_x, opt_y = opt(lgb, max_iters=10, train_x=x_train, train_y=y_train, cv_mun=10, init_start=5, model_score='neg_mean_squared_error', print_opt=True)

lgb_f = lgb.f(opt_x)
lgb_f.fit(x_train, y_train)
y_pre = lgb_f.predict(x_test)

endtime = time.time()
process_time = (endtime - start_time) * 1000

print('mse:', mean_squared_error(y_test, y_pre))

print('耗时：', process_time)


# 假设网络带宽：4800kb/s， 传输距离2000KM， 电信号传播速度光速的2/3   = 2e+5km/s
# 计算训练集、测试集数据的大小
size_train_data = sys.getsizeof(x_train) + sys.getsizeof(y_train)
size_test_data = sys.getsizeof(x_test) + sys.getsizeof(y_test)
# print('size_train_data:%s bit' %size_train_data, end=' ')
# print('size_test_data:%s bit' %size_test_data)
print('训练数据的延迟：', time_FS(data_len=size_train_data, speed_fs=4800e+3)*1000 + time_TB(dis_point=2e+3, speed_tb=2e+5)*1000)
print('测试数据的延迟：', time_FS(data_len=size_test_data, speed_fs=4800e+3)*1000 + time_TB(dis_point=2e+3, speed_tb=2e+5)*1000)
