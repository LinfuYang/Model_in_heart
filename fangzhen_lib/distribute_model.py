import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
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

list_mse =[]
for i in range(10):

    # 划分训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    x_train_1, x_train, y_train_1, y_train = train_test_split(x_train, y_train, test_size=0.25)
    x_train_2, x_train, y_train_2, y_train = train_test_split(x_train, y_train, test_size=1/3)
    x_train_3, x_train_4, y_train_3, y_train_4 = train_test_split(x_train, y_train, test_size=0.5)


    start_time = time.time()
    lin_m_1 = LinearRegression()
    # lin_m_1 = Lasso()
    lin_m_1.fit(x_train_1, y_train_1)

    lin_m_2 = LinearRegression()
    # lin_m_2 = Lasso()
    lin_m_2.fit(x_train_2, y_train_2)

    lin_m_3 = LinearRegression()
    # lin_m_3 = Lasso()
    lin_m_3.fit(x_train_3, y_train_3)

    lin_m_4 = LinearRegression()
    # lin_m_4 = Lasso()
    lin_m_4.fit(x_train_4, y_train_4)

    m_inter = []
    m_ceof = []

    # 数据聚合
    m_i = [lin_m_1, lin_m_2, lin_m_3, lin_m_4]
    for i in range(4):
        m_inter.append(m_i[i].intercept_)
        m_ceof.append(m_i[i].coef_)
    # 求均值
    # print(m_inter)
    # print(m_ceof )
    end_m_inter = np.mean(m_inter, axis=0)
    end_m_ceof =  np.mean(m_ceof, axis=0)
    # rint(end_m_inter)
    # rint(end_m_ceof )
    # 预测
    def predict_x(x_test_1, m_int, m_ce):
        ONE = np.array(m_ce, ndmin=2)
        two = np.array(x_test_1).T
        return np.reshape(m_int + np.dot(ONE, two), (-1, 1))

    y_pre = predict_x(x_test, end_m_inter, end_m_ceof)
# end_time = time.time()
# last_time = (end_time - start_time) * 1000
# print('进行一次聚合需要的时间', last_time)
    list_mse.append(mean_squared_error(y_test, y_pre))
print(list_mse)
print('聚合后的模型评价标准：', np.mean(list_mse))
