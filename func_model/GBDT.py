from sklearn.ensemble import GradientBoostingRegressor
import GPyOpt
import numpy as np

def f(x):
    clf_dt = GradientBoostingRegressor(
        learning_rate=x[0, 0],
        n_estimators=int(x[0, 1]),
        max_depth=int(x[0, 2]),
        min_samples_split=int(x[0, 3]),
    )

    return clf_dt

space = GPyOpt.Design_space(space=[{'name': 'learning_rate', 'type': 'continuous', 'domain': (10e-6, 1)},
                                   {'name': 'n_estimators', 'type': 'discrete', 'domain': tuple(range(5, 201))},
                                   {'name': 'max_depth', 'type': 'discrete', 'domain': tuple(range(2, 20))},
                                   {'name': 'min_samples_split', 'type': 'discrete', 'domain': tuple(range(2, 300))},

                                   ])