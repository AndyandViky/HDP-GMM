# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: cluster_process.py
@Time: 2020-03-14 15:45
@Desc: cluster_process.py
"""
import warnings
import numpy as np

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn._utils.compat import Memory
from nilearn.plotting import plot_prob_atlas, show, plot_stat_map
from nilearn.image import iter_img
from nilearn import datasets

from sklearn.mixture import GaussianMixture
from config import RESULT_DIR


class ClusterProcess(MultiPCA):

    def __init__(self, model, args=None, mask=None, group=4, n_components=20, smoothing_fwhm=6,
                 do_cca=True,
                 threshold='auto',
                 n_init=10,
                 random_state=None,
                 standardize=True, detrend=True,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='epi', mask_args=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        super(ClusterProcess, self).__init__(
            n_components=n_components,
            do_cca=do_cca,
            random_state=random_state,
            # feature_compression=feature_compression,
            mask=mask, smoothing_fwhm=smoothing_fwhm,
            standardize=standardize, detrend=detrend,
            low_pass=low_pass, high_pass=high_pass, t_r=t_r,
            target_affine=target_affine, target_shape=target_shape,
            mask_strategy=mask_strategy, mask_args=mask_args,
            memory=memory, memory_level=memory_level,
            n_jobs=n_jobs, verbose=verbose)

        self.model_ = model
        self.train_data = None
        self.pred = None
        self.group = group
        self.args = args

    def split_data(self, data):
        group = self.group
        labels = GaussianMixture(n_components=self.n_components).fit_predict(data)
        datas = list()
        category = np.unique(labels)
        for i in range(len(category)):
            datas.append(data[labels == i])
        index = int(len(category) / group)
        train_data = list()
        for i in range(group):
            j = np.random.randint(0, len(category), max(2, index * 2))
            train_data.append(np.vstack((datas[k] for k in j)))

        return train_data

    def model_fit(self, data):

        np.random.seed(0)

        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        self.train_data = data[np.random.randint(0, data.shape[0], 5000)]
        self.train_data = self.split_data(self.train_data)
        model = self.model_.fit(self.train_data)
        self.pred = model.predict(self.train_data)
        pred = model.predict(data)
        # self.pred = scio.loadmat('./datas/brain/adhd/pred/test_100.mat')['pred'].reshape(-1)
        data = data.T
        total = 0
        for i in range(self.n_components):
            data[i][pred != i] = 0
            data[i][pred == i] = 1
            total += data[i][data[i] != 0].shape[0]

        print(total)
        for component in data:
            if component.max() < -component.min():
                component *= -1
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(data)

        return self

    def _raw_fit(self, data):

        components = MultiPCA._raw_fit(self, data)
        self.model_fit(components)
        return self

    def plot(self, name='hdpvmf', save=False):

        components_img = self.components_img_
        warnings.filterwarnings("ignore")
        display = plot_prob_atlas(components_img, title='All components', view_type='filled_contours')

        # for i, cur_img in enumerate(iter_img(components_img)):
        #     plot_stat_map(cur_img, display_mode="z", title="Component %d" % i,
        #                   cut_coords=1, colorbar=False)
        show()
        if save:
            display.savefig('{}/adhd_result_{}.jpg'.format(RESULT_DIR, name), dpi=200)


if __name__ == '__main__':
    dataset = datasets.fetch_adhd(data_dir='./datas/brain', n_subjects=30)
    func_filenames = dataset.func
    cp = ClusterProcess(model=1, n_components=20, smoothing_fwhm=6.,
                          memory="nilearn_cache", memory_level=2,
                          threshold=3., verbose=10, random_state=0)
    cp.fit(func_filenames)
