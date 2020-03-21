# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: train_brain.py
@Time: 2020-03-16 22:47
@Desc: train_brain.py
"""

try:
    import argparse
    import numpy as np
    import torch
    import pandas as pd

    from model import VIModel
    from scipy import io as scio
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

    from config import DATA_PARAMS, BRAIN_DIR
    from utils import console_log, get_haxby_data, get_adhd_data
    from cluster_process import ClusterProcess

except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        if int(args.algorithm_category) == 0:
            self.model = VIModel(args)
        elif int(args.algorithm_category) == 1:
            self.model = SVIModel(args)
        elif int(args.algorithm_category) == 2:
            self.model = VIDP(args)
        else:
            pass

    def train(self, data):

        self.model.fit(data)


if __name__ == "__main__":

    # set args
    parser = argparse.ArgumentParser(prog='HIN-datas',
                                    description='Hierarchical Dirichlet process Mixture Models of datas Distributions')
    parser.add_argument('-c', '--algorithm_category', dest='algorithm_category', help='choose VIModel:0 or SVIModel:1',
                        default=0)
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='adhd')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    # hyper parameters
    parser.add_argument('-k', '--K', dest='K', help='truncation level K', default=12)
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=5)
    parser.add_argument('-z', '--zeta', dest='zeta', help='zeta', default=0.01)
    parser.add_argument('-u', '--u', dest='u', help='u', default=0.9)
    parser.add_argument('-v', '--v', dest='v', help='v', default=0.01)
    parser.add_argument('-tau', '--tau', dest='tau', help='top stick tau', default=1)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1)
    parser.add_argument('-th', '--threshold', dest='threshold', help='second threshold', default=1e-7)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.01)
    parser.add_argument('-sm', '--second_max_iter', dest='second_max_iter',
                        help='second max iteration of variational inference', default=-1)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=100)
    args = parser.parse_args()

    (K, T, mix_threshold, algorithm_category, max_iter, second_max_iter, threshold, group, dim, time, do_last_update,
     random_seed) = DATA_PARAMS[args.data_name]

    # train_data, test_data, nor_data, nor_test_data, labels, test_labels = get_haxby_data(subject_name=args.data_name, data_split=100,
    #                                                                                      random_seed=random_seed)
    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    if int(args.load_params) == 1:
        args.K = K
        args.T = T
        args.mix_threshold = mix_threshold
        args.algorithm_category = algorithm_category
        args.second_max_iter = second_max_iter
        args.threshold = threshold
        args.do_last_update = do_last_update
        args.max_iter = max_iter
    #
    # datas = list()
    # category = np.unique(labels)
    # for i in range(len(category)):
    #     datas.append(train_data[labels == i])
    # index = int(len(category) / group)
    # train_data = list()
    # for i in range(group):
    #     j = np.random.randint(0, len(category), max(2, index * 2))
    #     train_data.append(np.vstack((datas[k] for k in j)))
    #
    # # index = int(train_data.shape[0] / group)
    # # datas = list()
    # # for i in range(group):
    # #     if i != group - 1:
    # #         datas.append(train_data[i * index:(i + 1) * index])
    # #     else:
    # #         datas.append(train_data[i * index:])
    #
    # args.test_data = test_data
    # args.test_labels = test_labels
    # model = VIModel(args).fit(datas)
    # pred = model.predict(test_data)
    # ca = np.unique(pred)
    # print(ca)
    # console_log(pred=pred, labels=test_labels, model_name='-HDP-VMF-brain', newJ=model.newJ)

    n_cluster = 40
    func_filenames = get_adhd_data(data_dir='./datas/brain', n_subjects=30)
    cp = ClusterProcess(model=VIModel(args), n_components=n_cluster, smoothing_fwhm=6.,
                          memory="nilearn_cache", memory_level=2,
                          threshold=3., verbose=10, random_state=0)
    cp.fit(func_filenames)
    cp.plot(save=False, name='hdpgmm')
    pred = cp.pred
    # scio.savemat('./test_{}'.format(args.N * args.K), {'pred': pred})
    train_data = cp.train_data
    ca = np.unique(pred)
    print(ca)
    console_log(pred=pred, data=np.vstack((i for i in train_data)), model_name='HMM-VMF-brain')
