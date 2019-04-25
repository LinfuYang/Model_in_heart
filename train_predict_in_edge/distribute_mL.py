import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import time
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
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
x_train_1, x_train, y_train_1, y_train = train_test_split(x_train, y_train, test_size=0.25)
x_train_2, x_train, y_train_2, y_train = train_test_split(x_train, y_train, test_size=1/3)
x_train_3, x_train_4, y_train_3, y_train_4 = train_test_split(x_train, y_train, test_size=0.5)


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

print('聚合后的模型评价标准：', mean_squared_error(y_test, y_pre))



