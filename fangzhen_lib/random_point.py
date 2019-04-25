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
random_data = 40
x_data_1 = sample_point(round_xy=x_rpund, iter=random_data)
y_data_1 = [f_4d.f_obj(x_data[i]) for i in range(random_data)]

model = a_gpr.creat_gpr_model(x_data_1, np.reshape(y_data_1, (-1, 1)))

y_pre = [a_gpr.predict_mu_var(np.array(x_test[i], ndmin=2), model, re_var=False) for i in range(np.shape(x_test)[0])]

print(mean_squared_error(y_test, y_pre))
