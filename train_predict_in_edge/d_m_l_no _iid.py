import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
import random
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel('../data/cccp_data.xlsx')

# ['AT', 'V', 'AP', 'RH', 'PE']
column = ['AT', 'V', 'AP', 'RH']
for index in column:
    scaler = MinMaxScaler()
    data[index] = scaler.fit_transform((data[index].values).reshape(-1, 1))

x_data = data[['AT', 'V', 'AP', 'RH']].values
y_data = np.reshape(data[['PE']].values, (-1, 1))


# 划分训练集、测试集
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

slice = random.sample(list(range(len(x_train))), 500)
x_train_1 = [x_train[i] for i in slice]
y_train_1 = [y_train[i] for i in slice]

slice = random.sample(list(range(len(x_train))), 500)
x_train_2 = [x_train[i] for i in slice]
y_train_2 = [y_train[i] for i in slice]

slice = random.sample(list(range(len(x_train))), 500)
x_train_3 = [x_train[i] for i in slice]
y_train_3 = [y_train[i] for i in slice]

slice = random.sample(list(range(len(x_train))), 500)
x_train_4 = [x_train[i] for i in slice]
y_train_4 = [y_train[i] for i in slice]

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
    m_ceof.append(m_i[i].coef_[0])
# 求均值
end_m_inter = np.mean(m_inter, axis=0)
end_m_ceof =  np.mean(m_ceof, axis=0)

# 预测
def predict_x(x_test_1, m_int, m_ce):
    ONE = np.array(m_ce, ndmin=2)
    two = np.array(x_test_1).T
    return np.reshape(m_int + np.dot(ONE, two), (-1, 1))

y_pre = predict_x(x_test, end_m_inter, end_m_ceof)
end_time = time.time()
last_time = (end_time - start_time) * 1000
print('进行一次聚合需要的时间', last_time)
print('聚合后的模型评价标准：', mean_squared_error(y_test, y_pre))
