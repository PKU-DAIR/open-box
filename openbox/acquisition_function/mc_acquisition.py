# License: MIT

from typing import List

import numpy as np

from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.surrogate.base.base_model import AbstractModel
from openbox.surrogate.base.gp import GaussianProcess


class MCEI(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.long_name = 'MC-Expected Improvement'
        self.par = par
        self.eta = None
        self.mc_times = kwargs.get('mc_times', 10)

    def _compute(self, X: np.ndarray, **kwargs):
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        Y_samples = np.zeros(shape=(self.mc_times, len(X)))
        Y_samples[:, :] = self.model.sample_functions(X, n_funcs=self.mc_times).transpose()

        mc_ei = np.maximum(self.eta - Y_samples - self.par, 0)
        ei = mc_ei.mean(axis=0)
        ei = ei.reshape(-1, 1)
        return ei


class MCEIC(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 constraint_models: List[GaussianProcess],
                 par: float = 0.0,
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.long_name = 'MC-Expected Constrained Improvement'
        self.constraint_models = constraint_models
        self.par = par
        self.eta = None
        self.mc_times = kwargs.get('mc_times', 10)
        self.eps = kwargs.get('eps', 1)

    def _compute(self, X: np.ndarray, **kwargs):
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        Y_samples = np.zeros(shape=(self.mc_times, X.shape[0]))
        Y_samples[:, :] = self.model.sample_functions(X, n_funcs=self.mc_times).transpose()

        eic = np.maximum(self.eta - Y_samples - self.par, 0)
        for c_model in self.constraint_models:
            constraint_samples = np.zeros(shape=(self.mc_times, X.shape[0]))
            constraint_samples[:, :] = c_model.sample_functions(X, n_funcs=self.mc_times).transpose()
            eic *= 1/(1 + np.exp(constraint_samples/self.eps))

        eic = eic.mean(axis=0).reshape(-1, 1)
        return eic
