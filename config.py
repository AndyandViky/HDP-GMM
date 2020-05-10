# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: config.py
@Time: 2020-02-15 00:05
@Desc: config.py
"""

import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datas')
GENE_DIR = os.path.join(DATASETS_DIR, 'gene')
BRAIN_DIR = os.path.join(DATASETS_DIR, 'brain')

RESULT_DIR = os.path.join(REPO_DIR, 'result')

# difference datasets config
# K, T, mix_threshold, algorithm_category, max_iter, second_max_iter, threshold, group, dim, times
DATA_PARAMS = {
    'Sporulation': (56, 15, 0.01, 0, 22, -1, 1e-7, 50, 7, (0, 5, 15, 20, 25, 30, 35)),
    'yeast': (55, 15, 0.01, 0, 30, -1, 1e-7, 50, 7, (0, 9.5, 11.5, 13.5, 15.5, 18.5, 20.5)),
    'Human_Fibroblasts': (30, 10, 0.01, 0, 11, -1, 1e-7, 50, 18, [i*10 for i in range(18)]),
    'Spellman': (30, 10, 0.02, 0, 20, -1, 1e-7, 40, 18, [i*10 for i in range(18)]),

    # brain dataset has a additional parameter which is a random seed. / show the best score.
    '0_haxby': (12, 400, 0.01, 0, 20, -1, 1e-7, 4, 48, (), 1, 1),
    '1_haxby': (150, 15, 0.02, 0, 35, -1, 1e-7, 4, 48, (), 1, 6),
    '2_haxby': (150, 15, 0.01, 0, 35, -1, 1e-7, 4, 48, (), 1, 3),
    '3_haxby': (150, 15, 0.01, 0, 20, -1, 1e-7, 4, 48, (), 1, 2),
    '4_haxby': (150, 15, 0.01, 0, 95, -1, 1e-7, 4, 48, (), 1, 3),
    '5_haxby': (150, 15, 0.01, 0, 35, -1, 1e-7, 4, 48, (), 1, 3),

    'adhd': (150, 70, 0.01, 0, 10, 100, 1e-4, 30, 40, (), 1),
}
