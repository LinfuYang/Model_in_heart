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


for index in column_1:
    scaler = StandardScaler()
    data_train[index] = scaler.fit_transform((data_train[index].values).reshape(-1, 1))
for index in column_1:
    scaler = StandardScaler()
    data_test[index] = scaler.fit_transform((data_test[index].values).reshape(-1, 1))



# 划分训练集、测试集
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

x_train = data_train[column_1].values
x_test  = data_test[column_1].values

y_train = data_train['Appliances']
y_test = data_test['Appliances']

slice = random.sample(list(range(len(x_train))), 2000)
x_train_1 = [x_train[i] for i in slice]
y_train_1 = [y_train[i] for i in slice]

slice = random.sample(list(range(len(x_train))), 2000)
x_train_2 = [x_train[i] for i in slice]
y_train_2 = [y_train[i] for i in slice]

slice = random.sample(list(range(len(x_train))), 2000)
x_train_3 = [x_train[i] for i in slice]
y_train_3 = [y_train[i] for i in slice]

slice = random.sample(list(range(len(x_train))), 2000)
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
    m_ceof.append(m_i[i].coef_)



# 求均值
end_m_inter = np.mean(m_inter)
end_m_ceof =  np.mean(m_ceof, axis=0)

print(end_m_inter)
print(end_m_ceof)
# 预测
def predict_x(x_test_1, m_int, m_ce):
    ONE = np.array(m_ce, ndmin=2)
    two = np.array(x_test_1).T
    return np.reshape(m_int + np.dot(ONE, two), (-1, 1))

y_pre = predict_x(x_test, end_m_inter, end_m_ceof)
end_time = time.time()
last_time = (end_time - start_time) * 1000
print('进行一次聚合需要的时间', last_time)
print('聚合后的模型评价标准：', np.sqrt(mean_squared_error(y_test, y_pre)))