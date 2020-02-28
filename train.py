# -*- coding: utf-8 -*-
# @author: andy
# @contact: andy_viky@163.com
# @github: https://github.com/AndyandViky
# @csdn: https://blog.csdn.net/AndyViky
# @file: train.py
# @time: 2020/1/13 15:17
# @desc: train.py
try:
    import argparse
    import numpy as np
    import torch
    import pandas as pd

    from model import VIModel
    from scipy import io as scio
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

    from dpgmm import VDPGMM
    from config import DATA_PARAMS, DATASETS_DIR
    from utils import cluster_acc, console_log, get_data, plot_data

except ImportError as e:
    print(e)
    raise ImportError


class Trainer:

    def __init__(self, args):

        if int(args.algorithm_category) == 0:
            self.model = VIModel(args)
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
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='Human_Fibroblasts')
    parser.add_argument('-lp', '--load_params', dest='load_params', help='load_params', default=1)
    parser.add_argument('-verbose', '--verbose', dest='verbose', help='verbose', default=1)
    # hyper parameters
    parser.add_argument('-k', '--K', dest='K', help='truncation level K', default=6)
    parser.add_argument('-t', '--T', dest='T', help='truncation level T', default=50)
    parser.add_argument('-u', '--u', dest='u', help='u', default=1)
    parser.add_argument('-v', '--v', dest='v', help='v', default=1)
    parser.add_argument('-tau', '--tau', dest='tau', help='top stick tau', default=1)
    parser.add_argument('-gamma', '--gamma', dest='gamma', help='second stick gamma', default=1)

    parser.add_argument('-th', '--threshold', dest='threshold', help='second threshold', default=1e-7)
    parser.add_argument('-mth', '--mix_threshold', dest='mix_threshold', help='mix_threshold', default=0.01)
    parser.add_argument('-sm', '--second_max_iter', dest='second_max_iter',
                        help='second max iteration of variational inference', default=500)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=30)
    args = parser.parse_args()

    _, data, time_ser = get_data(DATASETS_DIR, args.data_name)
    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    K, T, mix_threshold, algorithm_category, max_iter, second_max_iter, threshold, group, dim, time = DATA_PARAMS[
        args.data_name]

    if int(args.load_params) == 1:
        args.K = K
        args.T = T
        args.mix_threshold = mix_threshold
        args.algorithm_category = algorithm_category
        args.second_max_iter = second_max_iter
        args.threshold = threshold

    args.max_iter = max_iter
    index = int(data.shape[0] / group)
    datas = list()
    for i in range(group):
        if i != group - 1:
            datas.append(data[i * index:(i + 1) * index])
        else:
            datas.append(data[i * index:])
    trainer = Trainer(args)
    trainer.train(datas)
    pred = trainer.model.predict(data)
    category = np.unique(np.array(pred))
    console_log(pred, data=data, model_name='===========hdp-gmm', each_data=[data[pred == i] for i in category],
                mu=trainer.model.mean_mu[category], newJ=trainer.model.newJ)




