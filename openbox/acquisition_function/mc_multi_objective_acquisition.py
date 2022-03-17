# License: MIT

from typing import List

import numpy as np
from scipy.stats import norm

from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.surrogate.base.base_model import AbstractModel
from openbox.surrogate.base.gp import GaussianProcess


class MCParEGO(AbstractAcquisitionFunction):
    def __init__(self,
                 model: List[AbstractModel],
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.long_name = 'Pareto Efficient Global Optimization'
        self.mc_times = kwargs.get('mc_times', 10)

    def _compute(self, X: np.ndarray, **kwargs):
        from openbox.utils.multi_objective import get_chebyshev_scalarization

        Y_samples = np.zeros(shape=(self.mc_times, X.shape[0], len(self.model)))
        for idx in range(len(self.model)):
            Y_samples[:, :, idx] = self.model[idx].sample_functions(X, n_funcs=self.mc_times).transpose()

        Y_mean = Y_samples.mean(axis=0)
        weights = np.random.random_sample(len(self.model))
        weights = weights / np.sum(weights)
        scalarized_obj = get_chebyshev_scalarization(weights, Y_mean)

        # Maximize the acq function --> Minimize the objective function
        acq = -scalarized_obj(Y_mean)
        acq = acq.reshape(-1, 1)
        return acq


class MCEHVI(AbstractAcquisitionFunction):
    r"""Monte Carlo Expected Hypervolume Improvement supporting m>=2 outcomes.

    This assumes minimization.

    Code is adapted from botorch. See [Daulton2020qehvi]_ for details.
    """

    def __init__(
        self,
        model: List[AbstractModel],
        ref_point,
        **kwargs
    ):
        """Constructor

        Parameters
        ----------
        model: A fitted model.
        ref_point: A list with `m` elements representing the reference point (in the
            outcome space) w.r.t. to which compute the hypervolume. This is a
            reference point for the objective values (i.e. after applying
            `objective` to the samples).
        """
        super().__init__(model=model, **kwargs)
        self.long_name = 'Monte Carlo Expected Hypervolume Improvement'
        self.mc_times = kwargs.get('mc_times', 10)
        ref_point = np.asarray(ref_point)
        self.ref_point = ref_point

    def _compute(self, X: np.ndarray, **kwargs):
        # Generate samples from posterior
        Y_samples = np.zeros(shape=(self.mc_times, X.shape[0], len(self.model)))
        for idx in range(len(self.model)):
            Y_samples[:, :, idx] = self.model[idx].sample_functions(X, n_funcs=self.mc_times).transpose()

        # Compute Y's hypervolume improvement by summing up contributions in each cell
        Z_samples = np.maximum(Y_samples, np.expand_dims(self.cell_lower_bounds, axis=(1, 2)))
        cubes = np.expand_dims(self.cell_upper_bounds, axis=(1, 2)) - Z_samples
        cubes[cubes < 0] = 0
        hvi = cubes.prod(axis=-1).sum(axis=0).mean(axis=0).reshape(-1, 1)
        return hvi


class MCParEGOC(MCParEGO):
    def __init__(self,
                 model: List[AbstractModel],
                 constraint_models: List[GaussianProcess],
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.long_name = 'Pareto Efficient Global Optimization with Constraints'
        self.eps = kwargs.get('eps', 1)

    def _compute(self, X: np.ndarray, **kwargs):
        acq = super()._compute(X)

        # Multiply by PoF (analytical)
        for c_model in self.constraint_models:
            m, v = c_model.predict_marginalized_over_instances(X)
            s = np.sqrt(v)
            acq *= norm.cdf(-m / s)

        # Multiplied by PoF (expectation of sigmoid approximation of indicator)
        # for c_model in self.constraint_models:
        #     constraint_samples = np.zeros(shape=(self.mc_times, X.shape[0]))
        #     constraint_samples[:, :] = c_model.sample_functions(X, n_funcs=self.mc_times).transpose()
        #     estimated_pof = 1/(1 + np.exp(constraint_samples/self.eps))
        #     estimated_pof = estimated_pof.mean(axis=0).reshape(-1, 1)
        #     acq *= estimated_pof

        return acq


class MCEHVIC(MCEHVI):
    r"""Monte Carlo Expected Hypervolume Improvement with constraints, supporting m>=2 outcomes.

    This assumes minimization.

    Code is adapted from botorch. See [Daulton2020qehvi]_ for details.

    """

    def __init__(
        self,
        model: List[AbstractModel],
        constraint_models: List[GaussianProcess],
        ref_point,
        **kwargs
    ):
        """Constructor

        Parameters
        ----------
        model: A fitted model.
        ref_point: A list with `m` elements representing the reference point (in the
            outcome space) w.r.t. to which compute the hypervolume. This is a
            reference point for the objective values (i.e. after applying
            `objective` to the samples).
        """
        super().__init__(model=model, ref_point=ref_point, **kwargs)
        self.long_name = 'Monte Carlo Expected Hypervolume Improvement with Constraints'
        self.eps = kwargs.get('eps', 1)

    def _compute(self, X: np.ndarray, **kwargs):
        acq = super()._compute(X)

        # Multiply by probability of feasibility
        for c_model in self.constraint_models:
            m, v = c_model.predict_marginalized_over_instances(X)
            s = np.sqrt(v)
            acq *= norm.cdf(-m / s)

        # Multiplied by PoF (expectation of sigmoid approximation of indicator)
        # for c_model in self.constraint_models:
        #     constraint_samples = np.zeros(shape=(self.mc_times, X.shape[0]))
        #     constraint_samples[:, :] = c_model.sample_functions(X, n_funcs=self.mc_times).transpose()
        #     estimated_pof = 1/(1 + np.exp(constraint_samples/self.eps))
        #     estimated_pof = estimated_pof.mean(axis=0).reshape(-1, 1)
        #     acq *= estimated_pof

        return acq
