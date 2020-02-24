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
    from sklearn.mixture import GaussianMixture

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
    parser.add_argument('-name', '--data_name', dest='data_name', help='data_name', default='big_data')
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
                        help='second max iteration of variational inference', default=-1)
    parser.add_argument('-m', '--max_iter', dest='max_iter', help='max iteration of variational inference', default=7)
    args = parser.parse_args()

    print('begin training......')
    print('========================dataset is {}========================'.format(args.data_name))

    data = scio.loadmat("{}/{}.mat".format(DATASETS_DIR, args.data_name))
    labels = data['z'].reshape(-1)
    data = data['data']

    test_data = data.reshape((-1, data[0].shape[1]))

    trainer = Trainer(args)
    trainer.train(data)
    pred = trainer.model.predict(data)
    # print(pred[:500])
    # print(np.unique(np.array(pred)))
    category = np.unique(np.array(pred))
    console_log(pred, data=test_data, labels=labels, model_name='===========hdp-gmm', each_data=[test_data[pred == i] for i in category],
                mu=trainer.model.mean_mu[category], newJ=trainer.model.newJ)

    # plot_data([time_ser[pred == i] for i in category], time, data_name=args.data_name, save=True)



