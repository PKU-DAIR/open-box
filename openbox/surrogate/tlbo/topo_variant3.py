# License: MIT

import numpy as np
from sklearn.model_selection import KFold
from openbox.surrogate.tlbo.base import BaseTLSurrogate
from openbox.surrogate.tlbo.scipy_solver import scipy_solve

_scale_method = 'standardize'


class TOPO_V3(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, fusion_method='idp_lc'):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'topo_1phase'
        self.fusion_method = fusion_method
        self.build_source_surrogates(normalize=_scale_method)
        # Weights for base surrogates and the target surrogate.
        self.w = np.array([1. / self.K] * self.K + [0.])
        self.base_predictions = list()
        self.min_num_y = 5

        self.hist_ws = list()
        self.iteration_id = 0
        self.target_y_range = None

    def batch_predict(self, X: np.ndarray):
        pred_y = None
        for i in range(0, self.K):
            mu, _ = self.source_surrogates[i].predict(X)
            if pred_y is not None:
                pred_y = np.c_[pred_y, mu]
            else:
                pred_y = mu
        return pred_y

    def predict_target_surrogate_cv(self, X, y):
        k_fold_num = 5
        _mu, _var = list(), list()

        # Conduct K-fold cross validation.
        kf = KFold(n_splits=k_fold_num)
        idxs = list()
        for train_idx, val_idx in kf.split(X):
            idxs.extend(list(val_idx))
            X_train, X_val, y_train, y_val = X[train_idx,:], X[val_idx,:], y[train_idx], y[val_idx]
            model = self.build_single_surrogate(X_train, y_train, normalize=_scale_method)
            mu, var = model.predict(X_val)
            mu, var = mu.flatten(), var.flatten()
            _mu.extend(list(mu))
            _var.extend(list(var))
        assert (np.array(idxs) == np.arange(X.shape[0])).all()
        return np.asarray(_mu), np.asarray(_var)

    def train(self, X: np.ndarray, y: np.array):
        instance_num = X.shape[0]
        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize=_scale_method)
        self.target_y_range = 0.5 * (np.max(y) - np.min(y))
        # print('Target y range', self.target_y_range)
        pred_y = self.batch_predict(X)

        w_new = None
        if instance_num < self.min_num_y:
            # Learn the weights of en problems.
            status, x = self.learn_source_weights(np.mat(pred_y), np.mat(y).T)
            if status:
                self.w[:self.K] = x
        else:
            # Learn the weights of all base surrogates.
            _pred_y, _ = self.predict_target_surrogate_cv(X, y)
            _pred_y = np.c_[pred_y, _pred_y.reshape((-1, 1))]
            status, x = self.learn_weights(np.mat(_pred_y), np.mat(y).T)
            w_source, w_target = x[:-1], x[-1]
            if status:
                w_target = np.max([w_target, 0.3])
                if instance_num >= 2 * self.min_num_y:
                    w_target = np.max([w_target, self.w[-1]])
                    w_source *= (1 - w_target)
            w_new = np.asarray(list(w_source) + [w_target])

            rho = 1.0
            self.w = rho * w_new + (1 - rho) * self.w

        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in w])
        print('In iter-%d' % self.iteration_id)
        print(weight_str)
        self.hist_ws.append(w)
        self.iteration_id += 1

    def learn_source_weights(self, pred_y, true_y):
        x, status = scipy_solve(pred_y, true_y, 3, debug=True)
        if status:
            x[x < 1e-3] = 0.
        return status, x

    def learn_weights(self, pred_y, true_y):
        x, status = scipy_solve(pred_y, true_y, 3, debug=True)
        if status:
            x[x < 1e-3] = 0.
        else:
            x = self.w.copy()
        return status, x

    def predict(self, X: np.array):
        return self.combine_predictions(X, self.fusion_method)

    def get_weights(self):
        return self.w
