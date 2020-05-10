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
import os
import numpy as np
import matplotlib.pyplot as plt

from nilearn.decomposition.multi_pca import MultiPCA
from nilearn._utils.compat import Memory
from nilearn.plotting import plot_prob_atlas, show, plot_stat_map
from nilearn.image import iter_img
from nilearn import datasets

from sklearn.mixture import GaussianMixture
from config import RESULT_DIR


class ClusterProcess(MultiPCA):

    def __init__(self, model, mask=None, n_components=20, smoothing_fwhm=6,
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
        self.model = None

    def hvmf_fit(self, data):

        self.train_data = data / np.linalg.norm(data, axis=2, keepdims=True)
        # self.train_data = (data - np.min(data, axis=1, keepdims=True)) / \
        #                   (np.max(data, axis=1) - np.min(data, axis=1))[:, np.newaxis]
        # scio.savemat('./datas/brain/adhd/test_sb3.mat', {'data': self.train_data})
        data = self.train_data[:, 1000:1500, :]
        # scio.savemat('./datas/brain/adhd/train_sb3.mat', {'data': data})

        self.model = self.model_.fit(data)
        # scio.savemat('./datas/brain/adhd/train_phi.mat', {'prior_': self.model.weights_, 'mu_': self.model.means_, 'Sigma_': self.model.covariances_})

        return self

    def _raw_fit(self, data):

        data = data.reshape((30, self.n_components, -1))

        # self.n_components = 40
        # datas = np.empty((30, self.n_components, data.shape[2]))
        # for i in range(30):
        #     datas[i] = MultiPCA._raw_fit(self, data[i]).T
        # plt.plot(data[0][:, :3])
        # plt.tight_layout()
        # plt.show()

        self.hvmf_fit(data.transpose((0, 2, 1)))
        return self

    def plot_pro(self, ita, save=False, item_file='group', name='vmf', choose=None, cut_coords=None, display_mode='ortho', keys=None):

        re_path = '{}/brain/{}/{}'.format(RESULT_DIR, name, item_file)
        if not os.path.exists(re_path):
            os.makedirs(re_path)

        for component in ita:
            if component.max() < -component.min():
                component *= -1
        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(ita)

        components_img = self.components_img_
        warnings.filterwarnings("ignore")
        display = plot_prob_atlas(components_img, title='All components', view_type='filled_contours')
        if save:
            display.savefig('{}/pro.png'.format(re_path), dpi=200)

        for i, cur_img in enumerate(iter_img(components_img)):
            if cut_coords is not None and display_mode is 'ortho':
                display = plot_stat_map(cur_img, cut_coords=cut_coords[i], display_mode=display_mode, dim=-.5,
                                        threshold=1e-2,
                                        cmap=plt.get_cmap('autumn'))
            elif display_mode is not 'ortho' and keys is not None:
                display = plot_stat_map(cur_img, cut_coords=cut_coords, display_mode=display_mode, dim=-.5,
                                        threshold=1e-2,
                                        cmap=plt.get_cmap('autumn'))
            else:
                display = plot_stat_map(cur_img, dim=-.5, display_mode=display_mode, threshold=1e-2,
                                        cmap=plt.get_cmap('autumn'))
            if save:
                if choose is not None:
                    display.savefig('{}/item{}.png'.format(re_path, choose[i] + 1), dpi=200)
                else:
                    display.savefig('{}/item{}.png'.format(re_path, i + 1), dpi=200)
        if save is False:
            show()

    def plot_all(self, pred, N=40, save=False, item_file='group', name='vmf'):

        data = np.zeros((N, pred.shape[0]))
        total = 0
        for i in range(N):
            data[i][pred != i] = 0
            data[i][pred == i] = 1
            total += data[i][data[i] != 0].shape[0]

        print(total)

        if hasattr(self, "masker_"):
            self.components_img_ = self.masker_.inverse_transform(data)

        components_img = self.components_img_
        warnings.filterwarnings("ignore")
        display = plot_prob_atlas(components_img, title='All components', view_type='filled_contours')
        if save:
            re_path = '{}/brain/{}/{}'.format(RESULT_DIR, name, item_file)
            if not os.path.exists(re_path):
                os.makedirs(re_path)
            display.savefig('{}/all.png'.format(re_path), dpi=200)
        else:
            show()


if __name__ == '__main__':
    dataset = datasets.fetch_adhd(data_dir='./datas/brain', n_subjects=30)
    func_filenames = dataset.func
    cp = ClusterProcess(model=1, n_components=20, smoothing_fwhm=6.,
                          memory="nilearn_cache", memory_level=2,
                          threshold=3., verbose=10, random_state=0)
    cp.fit(func_filenames)
