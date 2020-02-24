# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: utils.py
@Time: 2020-02-06 15:10
@Desc: utils.py
"""
import numpy as np
import warnings
import scipy.io as scio
import matplotlib.pyplot as plt

from config import RESULT_DIR

from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.special import iv
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI, \
        adjusted_mutual_info_score as AMI, adjusted_rand_score as AR, silhouette_score as SI, calinski_harabasz_score as CH


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w


def calculate_mix(a, b, K):

    lambda_bar = a / (a + b)
    pi = np.zeros(K,)
    for i in range(K):
        temp_temp = 1
        for j in range(i):
            temp = 1 - (lambda_bar[j])
            temp_temp = temp_temp * temp
        pi[i] = lambda_bar[i] * temp_temp
    return pi


def caculate_pi(model, N, T):

    resp = np.zeros((N, T))
    resp[np.arange(N), model.labels_] = 1
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    pi = (nk / N)[np.newaxis, :]
    return pi


def log_normalize(v):
    ''' return log(sum(exp(v)))'''
    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:, np.newaxis]

    return v, log_norm


def console_log(pred, data=None, labels=None, model_name='cluster', each_data=None, mu=None, newJ=None):

    measure_dict = dict()
    if data is not None:
        measure_dict['si'] = SI(data, pred)
        measure_dict['ch'] = CH(data, pred)
    if labels is not None:
        measure_dict['acc'] = cluster_acc(pred, labels)[0]
        measure_dict['nmi'] = NMI(labels, pred)
        measure_dict['ar'] = AR(labels, pred)
        measure_dict['ami'] = AMI(labels, pred)
    if each_data is not None and mu is not None:
        measure_dict['h_ave'] = compute_h_ave(each_data, mu, len(pred))[0]
        measure_dict['s_ave'] = compute_s_ave(each_data, mu)[0]
    if newJ is not None:
        measure_dict['new_component'] = newJ

    char = ''
    for (key, value) in measure_dict.items():
        char += '{}: {:.4f} '.format(key, value)
    print('{} {}'.format(model_name, char))


def compute_h_ave(data, mu, N):
    """
    The homogeneity measures
    data is a list, include j cluster
    :param data:
    :param mu:
    :param N:
    :return: h_ave, h_min
    """
    h_ave = 0
    h_min = list()
    for j, x in enumerate(data):
        ave = x.dot(mu[j][:, np.newaxis]) / (np.linalg.norm(x, axis=1) * np.linalg.norm(mu[j]))[:, np.newaxis]
        h_ave += np.sum(ave)

        h_min.append(np.min(ave))

    h_ave = h_ave / N
    return h_ave, h_min


def compute_s_ave(data, mu):
    """
    The separation measures
    data is a list, include j cluster
    :param data:
    :param mu:
    :return: s_ave, s_max
    """
    ave = 0
    ave_len = 0
    s_max = list()
    for i in range(len(data)):
        i_len = data[i].shape[0]
        i_u = mu[i][np.newaxis, :]
        for j in range(len(data)):
            if j == i:
                continue
            else:
                j_len = data[j].shape[0]
                u = (i_u.dot(mu[j][:, np.newaxis]) / (np.linalg.norm(i_u) * np.linalg.norm(mu[j])))[0][0]
                s_max.append(u)
                ave += i_len * j_len * u
                ave_len += i_len * j_len
    s_ave = ave / ave_len
    s_max = np.max(s_max)

    return s_ave, s_max


def get_data(data_dir, data_name):
    """
    yeast, Sporulation
    :param data_dir:
    :param data_name:
    :return: nor_data, data
    """
    datas = scio.loadmat('{}/{}.mat'.format(data_dir, data_name))
    time_ser = datas['data']
    data = 2 ** time_ser
    nor_data = data / np.sum(data, 1, keepdims=True)
    # labels = datas['z'].reshape(-1)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    return nor_data, data, time_ser


def plot_data(each_data, time, data_name=None, save=False):
    """
    plot different cluster graph
    :param each_data:
    :param time:
    :param save:
    :return: None
    """
    lens = len(each_data)
    fig, axs = plt.subplots(nrows=int(lens / 2) if lens % 2 == 0 else int(lens / 2 + 1), ncols=2, constrained_layout=False)

    for index, ax in enumerate(axs.flat):
        if lens % 2 != 0 and index == lens:
            fig.delaxes(ax)
            break
        ax.plot(time, each_data[index].T)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Expression', fontsize=12)
        ax.set_title('Cluster: {}'.format(index + 1), fontsize=14)
    plt.show()

    if save:
        fig.savefig('{}/cluster_result_{}.png'.format(RESULT_DIR, data_name))




