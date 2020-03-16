# -*- coding: utf-8 -*-
# @author: andy
# @contact: andy_viky@163.com
# @github: https://github.com/AndyandViky
# @csdn: https://blog.csdn.net/AndyViky
# @file: model.py
# @time: 2020/1/13 15:16
# @desc: model.py
try:
    import numpy as np
    import torch

    from scipy.special import digamma, iv, gammaln
    from sklearn.cluster import KMeans
    from numpy.matlib import repmat

    from utils import log_normalize, get_data, console_log
    from config import DATASETS_DIR
except ImportError as e:
    print(e)
    raise ImportError


class VIModel:
    """
    Variational Inference Hierarchical Dirichlet process Mixture Models of datas Distributions
    """
    def __init__(self, args):

        self.K = args.K
        self.T = args.T
        self.newJ = self.K
        self.second_max_iter = args.second_max_iter
        self.args = args
        self.J = 3
        self.N = 300
        self.D = 3
        self.prior = dict()

        self.tau = None
        self.gamma = None

        self.mean_mu = None
        self.cov_mu = None
        self.a_tao = None
        self.b_tao = None
        self.pi = None

        self.rho = None
        self.var_theta = None
        self.a = None
        self.b = None
        self.g = None
        self.h = None

        self.temp_top_stick = None
        self.temp_xi_ss = None
        self.temp_tao_ss = None
        self.temp_tao_b_ss = None

        self.det = 1e-10

    def _bound_x(self, X):
        N, D = X.shape
        bound_x = np.empty((self.newJ, N))
        for t in range(self.newJ):
            bound_x[t] = np.sum((X - self.mean_mu[t])**2, axis = 1) + np.trace(self.cov_mu[t])
        return bound_x

    def init_top_params(self, data):

        self.J = len(data)
        self.D = data[0].shape[1]

        # total_data = np.vstack((i for i in data))
        # self.mean_mu = KMeans(n_clusters=self.K).fit(total_data).cluster_centers_[::-1]
        self.mean_mu = np.zeros((self.K, self.D))
        self.cov_mu = np.empty((self.K, self.D, self.D))
        for i in range(self.K):
            self.cov_mu[i] = np.eye(self.D)

        self.a_tao = np.ones(self.K)
        self.b_tao = np.ones(self.K)

        self.prior = {
            'u': self.args.u,
            'v': self.args.v,
            'tau': self.args.tau,
            'gamma': self.args.gamma,
        }

        self.a = np.ones(self.K - 1)
        self.b = np.ones(self.K - 1)
        self.temp_top_stick = np.zeros(self.K)
        self.temp_xi_ss = np.zeros((self.K, self.D))
        self.temp_tao_ss = np.zeros(self.K)
        self.temp_tao_b_ss = None

        self.init_update(data)

    def set_temp_zero(self):

        self.temp_top_stick.fill(0.0)
        self.temp_xi_ss.fill(0.0)
        self.temp_tao_ss.fill(0.0)
        self.temp_tao_b_ss = None

    def init_update(self, x):

        self.var_theta = np.ones((self.T, self.K)) * (1 / self.K)

        for i in range(self.J):
            N = x[i].shape[0]
            self.rho = np.ones((N, self.T)) * (1 / self.T)
            self.temp_top_stick += np.sum(self.var_theta, 0)
            self.temp_tao_ss += np.sum(self.rho.dot(self.var_theta), 0)
            self.temp_xi_ss += self.var_theta.T.dot(self.rho.T.dot(x[i]))
            if i == 0:
                self.temp_tao_b_ss = self.rho.dot(self.var_theta)
            else:
                self.temp_tao_b_ss = np.vstack((self.temp_tao_b_ss, self.rho.dot(self.var_theta)))
        self.update_a_b()
        self.update_mu()
        self.update_tao(x)

    def calculate_new_com(self):

        threshold = self.args.mix_threshold

        index = np.where(self.pi > threshold)[0]
        self.pi = self.pi[self.pi > threshold]
        self.newJ = self.pi.size

        self.a_tao = self.a_tao[index]
        self.b_tao = self.b_tao[index]
        self.mean_mu = self.mean_mu[index]
        self.cov_mu = self.cov_mu[index]

        if self.args.verbose:
            print("new component is {}".format(self.newJ))

    def init_second_params(self, N):

        self.rho = np.ones((N, self.T)) * (1 / self.T)

        self.g = np.zeros(self.T - 1)
        self.h = np.zeros(self.T - 1)

        self.update_g_h(self.rho)

    def expect_log_sticks(self, a, b, k):

        sticks = np.zeros((2, k - 1))
        sticks[0] = a
        sticks[1] = b
        dig_sum = digamma(np.sum(sticks, 0))
        ElogW = digamma(sticks[0]) - dig_sum
        Elog1_W = digamma(sticks[1]) - dig_sum

        n = len(sticks[0]) + 1
        Elogsticks = np.zeros(n)
        Elogsticks[0:n - 1] = ElogW
        Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
        return Elogsticks

    def var_inf_2d(self, x, Elogsticks_1nd, j):

        Elog_phi = self._log_lik_x(self._bound_x(x)).T

        second_max_iter = 20000 if self.second_max_iter == -1 else self.second_max_iter
        self.init_second_params(x.shape[0])
        likelihood = 0.0
        old_likelihood = 1
        converge = 1
        Elogsticks_2nd = self.expect_log_sticks(self.g, self.h, self.T)
        for i in range(second_max_iter):
            # compute var_theta

            self.var_theta = self.rho.T.dot(Elog_phi) + Elogsticks_1nd
            log_var_theta, log_n = log_normalize(self.var_theta)
            self.var_theta = np.exp(log_var_theta)

            self.rho = self.var_theta.dot(Elog_phi.T).T + Elogsticks_2nd
            log_rho, log_n = log_normalize(self.rho)
            self.rho = np.exp(log_rho)

            self.update_g_h(self.rho)
            Elogsticks_2nd = self.expect_log_sticks(self.g, self.h, self.T)

            likelihood = 0.0
            # compute likelihood
            likelihood += np.sum((Elogsticks_1nd - log_var_theta) * self.var_theta)

            v = np.vstack((self.g, self.h))
            log_alpha = np.log(self.prior['gamma'])
            likelihood += (self.T - 1) * log_alpha
            dig_sum = digamma(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.prior['gamma']])[:, np.newaxis] - v) * (digamma(v) - dig_sum))
            likelihood -= np.sum(gammaln(np.sum(v, 0))) - np.sum(gammaln(v))

            # Z part
            likelihood += np.sum((Elogsticks_2nd - log_rho) * self.rho)

            # X part, the data part
            likelihood += np.sum(self.rho.T * np.dot(self.var_theta, Elog_phi.T))

            if i > 0:
                converge = (likelihood - old_likelihood) / abs(old_likelihood)
            old_likelihood = likelihood

            if converge < self.args.threshold:
                break

        self.temp_top_stick += np.sum(self.var_theta, 0)
        self.temp_tao_ss += np.sum(self.rho.dot(self.var_theta), 0)
        self.temp_xi_ss += self.var_theta.T.dot(self.rho.T.dot(x))
        if j == 0:
            self.temp_tao_b_ss = self.rho.dot(self.var_theta)
        else:
            self.temp_tao_b_ss = np.vstack((self.temp_tao_b_ss, self.rho.dot(self.var_theta)))

        return likelihood

    def var_inf(self, x):

        for ite in range(self.args.max_iter):

            self.set_temp_zero()
            Elogsticks_1nd = self.expect_log_sticks(self.a, self.b, self.K)
            for i in range(self.J):
                self.var_inf_2d(x[i], Elogsticks_1nd, i)

            self.update_a_b()
            self.update_mu()
            self.update_tao(x)

            # print(ite)
            pi = np.exp(self.expect_log_sticks(self.a, self.b, self.K))
            index = np.where(pi > self.args.mix_threshold)[0]
            print("ite: {}, index: {}".format(ite, index))

            if ite == self.args.max_iter - 1:
                # compute k
                self.pi = np.exp(self.expect_log_sticks(self.a, self.b, self.K))
                self.calculate_new_com()
                if self.args.verbose:
                    print('mu: {}'.format(self.mean_mu))
                    # print('con: {}'.format(self.con))
                    print('pi: {}'.format(self.pi))

    def _log_lik_x(self, bound_X):
        likx = np.zeros(bound_X.shape)
        for t in range(self.newJ):
            likx[t, :] = .5*self.D*(digamma(self.a_tao[t]) - np.log(self.b_tao[t]) - np.log(2*np.pi))
            tao_t = self.a_tao[t] / self.b_tao[t]
            likx[t, :] -= .5 * tao_t * bound_X[t]
        return likx

    def update_mu(self):
        for t in range(self.K):
            tao_t = self.a_tao[t] / self.b_tao[t]
            Nt = self.temp_tao_ss[t]
            self.cov_mu[t] = np.linalg.inv((tao_t * Nt + 1) * np.eye(self.D))
            self.mean_mu[t] = tao_t * self.cov_mu[t].dot(self.temp_xi_ss[t])

    def update_tao(self, x):
        bound = self._bound_x(np.vstack((i for i in x))).T
        temp = np.sum(bound * self.temp_tao_b_ss, 0)
        for t in range(self.K):
            self.a_tao[t] = self.prior['u'] + .5 * self.D * self.temp_tao_ss[t]
            self.b_tao[t] = self.prior['v'] + .5 * temp[t]

    def update_g_h(self, rho):
        # compute g, h
        self.g = 1 + np.sum(rho[:, :self.T - 1], 0)
        phi_cum = np.flipud(np.sum(rho[:, 1:], 0))
        self.h = self.prior['gamma'] + np.flipud(np.cumsum(phi_cum))

    def update_a_b(self):
        # compute a, b
        self.a = 1 + self.temp_top_stick[:self.K - 1]
        var_phi_sum = np.flipud(self.temp_top_stick[1:])
        self.b = self.prior['tau'] + np.flipud(np.cumsum(var_phi_sum))

    def fit(self, data):

        self.init_top_params(data)
        self.var_inf(data)

    def predict(self, data):
        # predict
        data = np.vstack((i for i in data))
        bound_X = self._bound_x(data)
        likc = np.log(self.pi)
        likx = self._log_lik_x(bound_X)
        s = likc[:, np.newaxis] + likx
        rho = np.exp(log_normalize(s.T)[0])

        pred = np.argmax(rho, axis=1)
        return pred

