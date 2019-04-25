from fangzhen_lib.sample_GPR import A_GPR
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
import time
from func_model.diatance_trance import time_FS, time_TB
import matplotlib.pyplot as plt
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
x_train, x_test, y_train, y_test = train_test_split(x_data, np.reshape(y_data, (-1, 1)), test_size=0.3)



m, n = np.shape(x_train)
print('训练数据总量', m)
print('低似真度数据样本量：', np.shape(x_train_l)[0])


a_gpr = A_GPR()




max_iter = 40
left = 40
right =41
dist = 50
list_arr = list(range(left, right, dist))


list_w = []
list_mse_gpr = []
start = time.time()
for ii in range(10):
    init_size = 2
    x_init_h, y_init_h = a_gpr.sample_point(x_data_conda=x_train, y_data_conda=np.array(y_train).reshape(-1, 1),
                                            iter=init_size, is_init=True)
    hf_gp, list_w_hf, y_pre = a_gpr.creat_gp_model(max_loop=max_iter, x_init_l=x_train_l, y_init_l=y_train_l,
                                                   x_init_h=x_init_h, y_init_h=y_init_h,
                                                   x_conda=np.array(x_train, ndmin=2),
                                                   y_conda=np.array(y_train).reshape(-1, 1),
                                                   list_iter=list_arr, data_test=x_test,

                                                   )
    list_w.append(list_w_hf)

    list_mse_gpr.append(mean_squared_error(y_test, y_pre[0]))


end = time.time()
# print(np.array(list_w))
print(np.mean(list_mse_gpr[0]))

plt.figure(figsize=(12, 8))
L_mean = np.mean(list_w, axis=0)
plt.plot(L_mean, lw=1.5, label='w-lh')
plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('w_hf')
plt.xlabel('iter')
plt.title('4D-case')
plt.show()
# print((end - start) * 1000)

# 假设网络带宽：4800b/s， 传输距离2000KM， 电信号传播速度光速的2/3   = 20Wkm/s

# 计算训练集、测试集数据的大小
size_train_data = sys.getsizeof(x_train) + sys.getsizeof(y_train)
size_test_data = sys.getsizeof(x_test) + sys.getsizeof(y_test)
# print('发送延迟：', time_FS(data_len=size_train_data, speed_fs=4800e+3)*1000)
# print('传播延迟：', time_TB(dis_point=2e+3, speed_tb=2e+5)*1000)