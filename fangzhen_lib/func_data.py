
from sklearn.model_selection import train_test_split
import numpy as np
from data.fangzhen_data.func_ND import func_4D_4


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
f_4d = func_4D_4()
x_rpund = f_4d.round_x

x_l = sample_point(round_xy=x_rpund, iter=50)
y_l = [f_4d.f_l(x_l[i, 0], x_l[i, 1], x_l[i, 2], x_l[i, 3]) for i in range(50)]


x_data = sample_point(round_xy=x_rpund, iter=5000)
y_data = [f_4d.f_obj(x_l[i, 0], x_l[i, 1], x_l[i, 2], x_l[i, 3]) for i in range(50000)]
x_train, x_test, y_train,y_test = train_test_split(x_data, y_data, test_size=0.3)



