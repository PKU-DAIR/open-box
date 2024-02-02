# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

import numpy as np

from openbox import logger
from openbox.utils.multi_objective import get_chebyshev_scalarization


class ParEGOSurrogate(object):
    def __init__(self, base_surrogate, seed):
        self.base_surrogate = base_surrogate 
        self.rng = np.random.RandomState(seed)
        self.scalarized_obj = None

    def train(self, X, Y):
        num_objectives = Y.shape[1]
        
        weights = self.rng.dirichlet(alpha=np.ones(num_objectives))
        logger.info(f'[ParEGO] Sampled weights: {weights}')
        self.scalarized_obj = get_chebyshev_scalarization(weights, Y)
        Y_scalarized = self.scalarized_obj(Y)

        self.base_surrogate.train(X, Y_scalarized)

    def predict(self, X):
        return self.base_surrogate.predict(X)

    def get_scalarized_obj(self):
        return self.scalarized_obj
    
    def predict_marginalized_over_instances(self, X):
        if hasattr(self.base_surrogate, "predict_marginalized_over_instances"):
            return self.base_surrogate.predict_marginalized_over_instances(X)
        else:
            raise NotImplementedError("predict_marginalized_over_instances is not implemented for the base surrogate.")

    def sample_functions(self, X, n_funcs=1):
        if hasattr(self.base_surrogate, "sample_functions"):
            return self.base_surrogate.sample_functions(X, n_funcs)
        else:
            raise NotImplementedError("Sampling functions is not implemented for the base surrogate.")
