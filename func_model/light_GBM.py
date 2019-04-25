
import lightgbm as lgb
import GPyOpt
import numpy as np



def f(x):

    clf_dt = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        learning_rate=x[0, 0],
        num_leaves=int(x[0, 1]),
        n_estimators=int(x[0, 2]),
        min_child_samples=int(x[0, 3]),
        max_depth=int(x[0, 4]),
        feature_fraction=x[0, 5],
        bagging_fraction=x[0, 6]


    )

    return clf_dt

space = GPyOpt.Design_space(space=[{'name': 'learning_rate', 'type': 'continuous', 'domain': (10e-6, 1)},
                                   {'name': 'num_leaves', 'type': 'discrete', 'domain': tuple(range(5, 100, 5))},
                                   {'name': 'n_estimators', 'type': 'discrete', 'domain': tuple(range(5, 201))},

                                   {'name': 'min_child_samples', 'type': 'discrete', 'domain': tuple(np.arange(2, 51))},

                                   {'name': 'max_depth', 'type': 'discrete', 'domain': tuple(range(2, 10))},
                                   {'name': 'feature_fraction', 'type': 'continuous', 'domain': (0.5, 1)},        # 建树的特征选择比例
                                   {'name': 'bagging_fraction', 'type': 'continuous', 'domain': (0.5, 1)},        # # 建树的样本采样比例

                                   ])



def pre_dict_model(x, x_train=None, y_train=None, x_test=None):

    clf = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        learning_rate=x[0, 0],
        num_leaves=int(x[0, 1]),
        n_estimators=int(x[0, 2]),
        min_child_samples=int(x[0, 3]),
        max_depth=int(x[0, 4]),
        feature_fraction=x[0, 5],
        bagging_fraction=x[0, 6]

    )
    clf.fit(x_train, y_train)
    importance = clf.feature_importance()

    # print('importtance:', importance)


    y_pre = clf.predict(x_test)

    return y_pre