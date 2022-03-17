# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

import numpy as np
from openbox.surrogate.base.base_model import AbstractModel
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances


class RandomForestEnsemble(AbstractModel):
    def __init__(self, types: np.ndarray,
                 bounds: np.ndarray, s_max, eta, weight_list, fusion_method, **kwargs):
        super().__init__(types=types, bounds=bounds, **kwargs)

        self.s_max = s_max
        self.eta = eta
        self.fusion = fusion_method
        self.surrogate_weight = dict()
        self.surrogate_container = dict()
        self.surrogate_r = list()
        self.weight_list = weight_list
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.surrogate_r.append(r)
            self.surrogate_weight[r] = self.weight_list[index]
            self.surrogate_container[r] = RandomForestWithInstances(types=types, bounds=bounds)

    def train(self, X: np.ndarray, Y: np.ndarray, r) -> 'AbstractModel':
        """Trains the Model on X and Y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, n_objectives]
            The corresponding target values. n_objectives must match the
            number of target names specified in the constructor.
        r : int
            Determine which surrogate in self.surrogate_container to train.

        Returns
        -------
        self : AbstractModel
        """
        self.types = self._initial_types.copy()

        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != len(self.types):
            raise ValueError('Feature mismatch: X should have %d features, but has %d' % (X.shape[1], len(self.types)))
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X.shape[0] (%s) != y.shape[0] (%s)' % (X.shape[0], Y.shape[0]))

        self.n_params = X.shape[1] - self.n_feats

        # reduce dimensionality of features of larger than PCA_DIM
        if self.pca and X.shape[0] > self.pca.n_components:
            X_feats = X[:, -self.n_feats:]
            # scale features
            X_feats = self.scaler.fit_transform(X_feats)
            X_feats = np.nan_to_num(X_feats)  # if features with max == min
            # PCA
            X_feats = self.pca.fit_transform(X_feats)
            X = np.hstack((X[:, :self.n_params], X_feats))
            if hasattr(self, "types"):
                # for RF, adapt types list
                # if X_feats.shape[0] < self.pca, X_feats.shape[1] ==
                # X_feats.shape[0]
                self.types = np.array(
                    np.hstack((self.types[:self.n_params], np.zeros((X_feats.shape[1])))),
                    dtype=np.uint,
                )

        return self._train(X, Y, r)

    def _train(self, X: np.ndarray, y: np.ndarray, r):
        self.surrogate_container[r].train(X, y)

    def _predict(self, X: np.ndarray):
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.types.shape[0], X.shape[1]))
        if self.fusion == 'idp':
            means, vars = np.zeros((X.shape[0], 1)), np.zeros((X.shape[0], 1))
            for r in self.surrogate_r:
                mean, var = self.surrogate_container[r].predict(X)
                means += self.surrogate_weight[r] * mean
                vars += self.surrogate_weight[r] * self.surrogate_weight[r] * var
            return means.reshape((-1, 1)), vars.reshape((-1, 1))
        else:
            raise ValueError('Undefined Fusion Method: %s!' % self.fusion)
