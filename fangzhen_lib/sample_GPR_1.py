import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared
# from GPyOpt.acquisitions import AcquisitionLCB
import GPy
from scipy import stats
from scipy.spatial.distance import pdist
class A_GPR_1:

    def __init__(self, f_kernel=None):
        if f_kernel != None:
            self.kernel_f = f_kernel

    def creat_gpr_model(self, x_data, y_data):
        '''
        :param x_data:  初始化的自变量
        :param y_data:  初始化的函数值
        :return: 新建的模型
        '''
        # gp = GaussianProcessRegressor(normalize_y=False)
        # gp.fit(x_data, y_data)
        m, n = np.shape(x_data)
        k_RBF = GPy.kern.RBF(input_dim=n, variance=0.5, lengthscale=0.8)
        gp = GPy.models.GPRegression(x_data, y_data, kernel=k_RBF)
        gp.optimize(messages=False)
        return gp

    def sample_point(self, x_data_conda=None, y_data_conda=None, iter=None, haved_point=None, is_init=False):
        m, n = np.shape(x_data_conda)
        x_temp = np.zeros((iter, n))
        y_temp = np.zeros((iter, 1))

        if is_init:
            sample_list = random.sample(list(range(m)), iter)

            for k in range(iter):
                x_temp[k] = x_data_conda[sample_list[k]]
                y_temp[k] = y_data_conda[sample_list[k]]
        else:
            l_have = list(haved_point)
            l_cond = list(x_data_conda)
            list_l = []
            for i in range(len(l_have)):
                for j in range(len(l_cond)):
                    if list(l_have[i]) == list(l_cond[j]):
                        list_l.append(j)

            x_conda_point = []
            y_conda_point = []
            for i in range(np.shape(x_data_conda)[0]):
                if i not in list_l:
                    x_conda_point.append(x_data_conda[i])
                    y_conda_point.append(y_data_conda[i])
            m_conda = np.shape(x_conda_point)[0]
            print(m_conda)
            sample_list = random.sample(list(range(m_conda)), iter)
            # print('sample_list', sample_list)
            for k in range(iter):
                x_temp[k] = x_conda_point[sample_list[k]]
                y_temp[k] = y_conda_point[sample_list[k]]

        return x_temp, y_temp

    def predict_mu_var(self, x_new, model_gp, re_var=True):
        '''
        :param x_new: 需要做出预测的点
        :param x_lf: 已知低质量点
        :param y_lf: 低质量点的对应函数值
        :return: 关于x_new的预测值 均值和方差
        '''
        # 求L-F dataset 的GP 均值 和  协方差矩阵
        # print(gp_lf._y_train_mean)
        # print(gp_lf.tianjia_k)
        # 输出任意自变量点的均值和标准差
        if re_var:

            # y_mean_lf, y_std_lf = model_gp.predict(x_new, return_std=re_var)
            y_mean_lf, y_var_lf = model_gp.predict(x_new)

            return y_mean_lf[0][0], y_var_lf[0][0]
        else:
            # y_mean_lf = model_gp.predict(x_new, return_std=re_var)
            y_mean_lf, _ = model_gp.predict(x_new)
            return y_mean_lf[0][0]

    def find_next_point(self, model_gp_hf=None, mode_gp_lf=None, lf_w=None, hf_w=None, con_data_x=None, con_data_y=None, restar_iter=20, single_iter=200, haved_point=None):
        '''
        :param model_gp:
        :param lf_w:
        :param hf_w:
        :param restar_iter:
        :param single_iter:
        :return:
        '''
        # print('find the next point')
        cond_x = []
        cond_y = []
        cond_confi = []
        for i in range(restar_iter):
            point_x, point_y = self.sample_point(x_data_conda=con_data_x, y_data_conda=con_data_y, iter=single_iter, haved_point=haved_point)


            temp_best = []
            for j in range(single_iter):
                if len(point_x[j]) == 1:
                    x_temp = np.array(point_x[j]).reshape(-1, 1)
                else:
                    x_temp = np.array(point_x[j]).reshape(1, -1)
                lf_mu, lf_var = self.predict_mu_var(x_temp, mode_gp_lf)
                hf_mu, hf_var = self.predict_mu_var(x_temp, model_gp_hf)
                p_1 = hf_var ** (-1)
                p_2 = lf_var ** (-1)
                post_mu = (hf_mu * hf_w * p_1 + lf_mu * lf_w * p_2) / (hf_w * p_1 + lf_w * p_2)
                post_cov = (hf_w * p_1 + lf_w * p_2) ** (-1)

                # 为找的下一个合适的采样点：
                f = 2 * post_cov ** 0.5
                temp_best.append(f)
            # print('每次迭代的结果', temp_best)
            s_best_index = np.argmax(temp_best)

            cond_confi.append(temp_best[s_best_index])
            cond_x.append(point_x[s_best_index])
            cond_y.append(point_y[s_best_index])
        # print('历次迭代的最佳值', cond_y)

        f_best_index = np.argmax(cond_confi)

        if len(cond_x[f_best_index]) == 1:
            next_point_x = np.array(cond_x[f_best_index]).reshape(-1, 1)
        else:
            next_point_x = np.array(cond_x[f_best_index]).reshape(1, -1)
        next_point_y = np.array(cond_y[f_best_index]).reshape(-1, 1)

        return next_point_x, next_point_y

    def find_next_point_2(self, model_gp_hf=None, mode_gp_lf=None, lf_w=None, hf_w=None, round_1D=None, x_init=None, single_iter=1000):
        '''
        :param model_gp:
        :param lf_w:
        :param hf_w:
        :param restar_iter:
        :param single_iter:
        :return:
        '''

        distance_arr = np.zeros((x_init.shape[0], single_iter))

        point_array = self.sample_point(round_1D, iter=single_iter)

        for i in range(x_init.shape[0]):
            for j in range(single_iter):
                distance_arr[i, j] = self.Eu_dist(x_init[i], point_array[j])
        best_point = np.argmax(distance_arr)
        # print(distance_arr)
        row = best_point // single_iter
        col = best_point - row * single_iter

        next_point = np.array(point_array[col], ndmin=2)


        '''
                lf_mu, lf_var = self.predict_mu_var(x_temp, mode_gp_lf)
                hf_mu, hf_var = self.predict_mu_var(x_temp, model_gp_hf)
                p_1 = hf_var ** (-1)
                p_2 = lf_var ** (-1)
                post_mu = (hf_mu * hf_w * p_1 + lf_mu * lf_w * p_2) / (hf_w * p_1 + lf_w * p_2)
                post_cov = (hf_w * p_1 + lf_w * p_2) ** (-1)

                # 为找的下一个合适的采样点：
                f = 2 * post_cov ** 0.5
                temp_best.append(f)
            # print('每次迭代的结果', temp_best)
            s_best_index = np.argmax(temp_best)

            cond_y.append(temp_best[s_best_index])
            cond_x.append(point_array[s_best_index])
        # print('历次迭代的最佳值', cond_y)

        f_best_index = np.argmax(cond_y)

        if len(cond_x[f_best_index]) == 1:
            next_point = np.array(cond_x[f_best_index]).reshape(-1, 1)
        else:
            next_point = np.array(cond_x[f_best_index]).reshape(1, -1)
        '''

        return next_point


    def like_hood_func(self, y_pre, mu, var):
        one = pow((2 * np.pi * var), 0.5)
        two = np.exp(-1 * (y_pre - mu) ** 2 / (2 * var)) + 10e-6

        return 1 / one * two

    def Eu_dist(self, x_1, x_2):

        return pdist(np.vstack([x_1, x_2]))

    def creat_gp_model(self, max_loop=15, x_init_l=None, y_init_l=None, x_init_h=None, y_init_h=None, n_start = 1, n_single=200, x_conda=None, y_conda=None, list_iter=None, data_test=None):
        '''

        :param max_loop:
        :param func_nd:
        :param a_gpr:
        :param x_init_l:
        :param y_init_l:
        :param x_init_h:
        :param y_init_h:
        :param round_x:
        :return:
        '''
        # 初始化模型

        lf_gp = self.creat_gpr_model(x_init_l, y_init_l)
        hf_gp = self.creat_gpr_model(x_init_h, y_init_h)
        print('模型初始化完毕')

        # 定义初始权重
        w_lf = 0.5
        w_hf = 1 - w_lf
        list_w_hf = list([w_hf])
        y_pre_list = []
        for it in range(3, max_loop+1):

            next_point_x, next_point_y = self.find_next_point(model_gp_hf=hf_gp, mode_gp_lf=lf_gp, lf_w=w_lf, hf_w=w_hf,
                                                         con_data_x=x_conda, con_data_y=y_conda,
                                                         restar_iter=n_start, single_iter=n_single, haved_point=x_init_h)

            # 检查该点是否已存在于历史数据中
            flag = True

            while flag:
                temp_flag = 0
                for l_d in x_init_h:
                    if list(l_d) == list(next_point_x[0]):
                        temp_flag = 1
                        print('temp_flag:', temp_flag)
                        print(l_d)
                        print(next_point_x[0])
                        break

                if temp_flag == 1:
                        next_point_x, next_point_y = self.find_next_point(model_gp_hf=hf_gp, mode_gp_lf=lf_gp, lf_w=w_lf, hf_w=w_hf,
                                                             con_data_x=x_conda, con_data_y=y_conda,
                                                             restar_iter=n_start, single_iter=n_single, haved_point=x_init_h)
                else:
                    flag = False

            # 计算w_lf的先验预估

            w_lf_pre = (w_lf**0.9) / (w_lf**0.9 + (1 - w_lf**0.9))
            w_hf_pre = 1 - w_lf_pre
            # 计算似然

            # print('next_point_x:', next_point_x, end=' ')
            # print('next_point_y:', next_point_y)

            mu_next_point_lf, var_next_point_lf = self.predict_mu_var(next_point_x, lf_gp)
            mu_next_point_hf, var_next_point_hf = self.predict_mu_var(next_point_x, hf_gp)
            print('mu_next_point_lf:', mu_next_point_lf, end=' ')
            print('cov_next_point_lf:', var_next_point_lf)
            print('mu_next_point_hf:', mu_next_point_hf, end=' ')
            print('cov_next_point_hf:', var_next_point_hf)
            print('next_point_y:', next_point_y[0, 0])
            like_hood_lf = round(stats.norm.pdf(next_point_y[0, 0], mu_next_point_lf, var_next_point_lf ** 0.5), 10) + 10e-10
            like_hood_hf = round(stats.norm.pdf(next_point_y[0, 0], mu_next_point_hf, var_next_point_hf ** 0.5), 10) + 10e-10
            print('like_hood_lf:', like_hood_lf, end=' ')
            print('like_hood_hf:', like_hood_hf)
            # like_hood_lf_1 = a_gpr.like_hood_func(y_pre_next_point, mu_next_point_lf, var_next_point_lf)
            # like_hood_hf_1 = a_gpr.like_hood_func(y_pre_next_point, mu_next_point_hf, var_next_point_hf)
            # print('like_hood_lf_1:', like_hood_lf_1, end=' ')
            # print('like_hood_hf_1:', like_hood_hf_1)

            dist_list = [self.Eu_dist(x_init_h[k], next_point_x) for k in range(x_init_h.shape[0])]
            # dist_list_l = [self.Eu_dist(x_init_l[k], next_point_x) for k in range(x_init_l.shape[0])]
            print('the min diatance', np.min(dist_list))
            # print('the min diatance x_nit_l:', np.min(dist_list_l))
            if np.min(dist_list) > 0.04:
                w_lf_post = w_lf_pre * like_hood_lf / (w_lf_pre * like_hood_lf + w_hf_pre * like_hood_hf)
                w_lf = w_lf_post
                # max_point_y = y_pre_next_point
                # max_point_x = next_point_x
            else:
                w_lf = w_lf

            if w_lf == 0.0:
                w_lf =30e-5
            w_hf = 1 - w_lf
            if w_hf == 0:
                w_hf =30e-5
            w_lf = 1 - w_hf
            list_w_hf.append(w_hf)

            x_init_h = np.r_[x_init_h, next_point_x]
            y_init_h = np.r_[y_init_h, next_point_y]
            print('x_init_h的规模:', np.shape(x_init_h), end=' ')
            print('w_hf:', w_hf)
            hf_gp = self.creat_gpr_model(x_init_h, y_init_h)

            if it in list_iter:
                y_temp_pre = []
                for ii in data_test:
                    y_temp_pre.append(self.h_f_pre(ii, lf_gp, hf_gp, w_lf, w_hf))
                '''
                # print(data_test[0])
                # 1
                x_test_1 = data_test[1]
                x_test_1[2] = y_temp_pre[0]
                # print(x_test_1)
                y_temp_pre.append(self.h_f_pre(x_test_1, lf_gp, hf_gp, w_lf, w_hf))
                # 2
                x_test_2 = data_test[2]
                x_test_2[2] = y_temp_pre[1]
                x_test_2[3] = y_temp_pre[0]
                y_temp_pre.append(self.h_f_pre(x_test_2, lf_gp, hf_gp, w_lf, w_hf))
                # print(x_test_2)
                d_NSM = data_test[:, 0]
                for i in range(3, np.shape(data_test)[0]):

                    x_test_i = data_test[i]
                    if d_NSM[i] - d_NSM[i - 1] == 1:
                        x_test_i[2] = y_temp_pre[i - 1]
                    else:
                        x_test_i[2] = 0
                    if d_NSM[i] - d_NSM[i - 2] == 2:
                        x_test_i[3] = y_temp_pre[i - 2]
                    else:
                        x_test_i[3] = 0
                    if d_NSM[i] - d_NSM[i - 3] == 3:
                        x_test_i[4] = y_temp_pre[i - 3]
                    else:
                        x_test_i[4] = 0
                    # print(x_test_i)
                    y_temp_pre.append(self.h_f_pre(x_test_i, lf_gp, hf_gp, w_lf, w_hf))
                y_pre_list.append(y_temp_pre)
                '''

                y_pre_list.append(y_temp_pre)

        return hf_gp, list_w_hf, y_pre_list

    def h_f_pre(self, point, m_lf, m_hf, wlf, whf):

        temp_x = np.array(point).reshape(1, -1)
        mu_temp_l, var_temp_l = self.predict_mu_var(temp_x, m_lf)
        mu_temp_h, var_temp_h = self.predict_mu_var(temp_x, m_hf)

        p_1 = var_temp_h ** (-1)
        p_2 = var_temp_l ** (-1)

        post_mu = (mu_temp_h * whf * p_1 + mu_temp_l * wlf * p_2) / (whf * p_1 + wlf * p_2)

        return post_mu