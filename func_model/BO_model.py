import GPyOpt
# from sklearn.externals import joblib
import sklearn.model_selection
import numpy as np

import sklearn
def opt(algo=None, max_iters=None, train_x=None, train_y=None, cv_mun=10, init_start=5,model_score=None,
        print_opt=False, smp=True):
    '''
    :param algo:  算法模型
    :param max_iters: 最大迭代此
    :param train_x: 训练特征
    :param train_y: 训练集标签
    :param cv_mun: 交叉验证次数
    :param init_start: 初始采样个数
    :param type_score: 模型评价标准
    :return:
    '''

    model_list = []
    def func_model(x):

        clf_dt = algo.f(x)
        scores = -np.mean(
            sklearn.model_selection.cross_val_score(clf_dt, train_x, train_y, cv=cv_mun, scoring=model_score))
        if print_opt:
            print('X:', x, end='\t\t')
            print('Y:', scores)

        return scores

    space = algo.space
    model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False)
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    initial_design = GPyOpt.experiment_design.initial_design('random', space, init_start)
    acquisition = GPyOpt.acquisitions.EI.AcquisitionEI(model, optimizer=aquisition_optimizer, space=space)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    objective = GPyOpt.core.task.SingleObjective(func_model)
    bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design,
                                                    de_duplication=True)
    bo.run_optimization(max_iter=max_iters, verbosity=False, save_models_parameters=smp)
    a, b = bo.get_evaluations()
    best_index = np.argmin(b)
    best_x = np.reshape(a[best_index], (1, -1))
    best_y = b[best_index]

    return best_x, best_y