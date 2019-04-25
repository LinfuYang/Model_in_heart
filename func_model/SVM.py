from sklearn.svm import SVR
import GPyOpt
import numpy as np

def f(x):

    clf_dt = SVR(
        gamma=x[0, 0],
        C=x[0, 1]

    )
    return clf_dt
space = GPyOpt.Design_space(space=[{'name': 'gamma', 'type': 'continuous', 'domain': (10e-6, 10)},
                                   {'name': 'C', 'type': 'continuous', 'domain': (10e-6, 500)},        # 建树的特征选择比例

                                   ])



def pre_dict_model(x, x_train=None, y_train=None, x_test=None):

    clf = SVR(
        gamma=x[0, 0],
        C=x[0, 1]

    )
    clf.fit(x_train, y_train)
    importance = clf.feature_importance()

    # print('importtance:', importance)


    y_pre = clf.predict(x_test)

    return y_pre