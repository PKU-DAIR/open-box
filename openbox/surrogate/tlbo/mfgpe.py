# License: MIT

import numpy as np
from typing import List
from collections import OrderedDict
from sklearn.model_selection import KFold
from openbox.surrogate.tlbo.base import BaseTLSurrogate

_scale_method = 'standardize'


class MFGPE(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed,
                 surrogate_type='rf', num_src_hpo_trial=-1, only_source=False):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'mfgpe'
        self.only_source = only_source
        self.build_source_surrogates(normalize=_scale_method)

        self.scale = True

        self.K = 0
        self.w = None
        self.snapshot_w = None
        self.hist_ws = list()
        self.iteration_id = 0

    def update_mf_trials(self, mf_hpo_data: List[OrderedDict]):
        if self.K == 0:
            self.K = len(mf_hpo_data) - 1  # K is the number of low-fidelity groups
            self.w = [1. / self.K] * self.K + [0.]
            self.snapshot_w = self.w
        self.source_hpo_data = mf_hpo_data
        # Refit the base surrogates.
        self.build_source_surrogates(normalize=_scale_method)

    def predict_target_surrogate_cv(self, X, y):
        k_fold_num = 5
        _mu, _var = list(), list()

        # Conduct K-fold cross validation.
        kf = KFold(n_splits=k_fold_num)
        idxs = list()
        for train_idx, val_idx in kf.split(X):
            idxs.extend(list(val_idx))
            X_train, X_val, y_train, y_val = X[train_idx, :], X[val_idx, :], y[train_idx], y[val_idx]
            model = self.build_single_surrogate(X_train, y_train, normalize=_scale_method)
            mu, var = model.predict(X_val)
            mu, var = mu.flatten(), var.flatten()
            _mu.extend(list(mu))
            _var.extend(list(var))
        assert (np.array(idxs) == np.arange(X.shape[0])).all()
        return np.asarray(_mu), np.asarray(_var)

    def train(self, X: np.ndarray, y: np.array, **kwargs):
        snapshot_weight = kwargs.get('snapshot', True)
        if snapshot_weight:
            self.w = self.snapshot_w
        sample_num = y.shape[0]

        if self.source_hpo_data is None:
            raise ValueError('Source HPO data is None!')

        # Get the predictions of low-fidelity surrogates
        mu_list, var_list = list(), list()
        for id in range(self.K):
            mu, var = self.source_surrogates[id].predict(X)
            mu_list.append(mu.flatten())
            var_list.append(var.flatten())

        # Evaluate the generalization of the high-fidelity surrogate via CV
        if sample_num >= 5:
            _mu, _var = self.predict_target_surrogate_cv(X, y)
            mu_list.append(_mu)
            var_list.append(_var)
            self.w = self.get_w_ranking_pairs(mu_list, var_list, y)

        if snapshot_weight:
            self.snapshot_w = self.w
            weight_str = ','.join([('%.2f' % item) for item in self.snapshot_w])
            self.logger.info('In iter-%d' % self.iteration_id)
            self.target_weight.append(self.snapshot_w[-1])
            self.logger.info('Weight: ' + str(weight_str))
            self.iteration_id += 1
            self.hist_ws.append(self.snapshot_w)

    def get_w_ranking_pairs(self, mu_list, var_list, y_true):
        preserving_order_p, preserving_order_nums = list(), list()
        for i in range(self.K + 1):
            y_pred = mu_list[i]
            preorder_num, pair_num = self.calculate_preserving_order_num(y_pred, y_true)
            preserving_order_p.append(preorder_num / pair_num)
            preserving_order_nums.append(preorder_num)
        n_power = 3
        trans_order_weight = np.array(preserving_order_p)
        p_power = np.power(trans_order_weight, n_power)
        return p_power / np.sum(p_power)

    def predict(self, X: np.array):
        sample_num = X.shape[0]
        mu, var = np.zeros((sample_num, 1)), np.zeros((sample_num, 1))

        # Base surrogate predictions with corresponding weights.
        for i in range(self.K + 1):
            mu_t, var_t = self.source_surrogates[i].predict(X)
            mu += self.w[i] * mu_t
            var += self.w[i] * self.w[i] * var_t
        return mu, var

    def get_weights(self):
        return self.w

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if not ((y_true[idx] > y_true[inner_idx]) ^ (y_pred[idx] > y_pred[inner_idx])):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num
