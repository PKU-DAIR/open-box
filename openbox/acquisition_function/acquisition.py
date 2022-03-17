# License: MIT
# This file is partially built on SMAC3(https://github.com/automl/SMAC3), which is licensed as follows,

# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

# encoding=utf8

import abc
import logging
from typing import List, Union

import numpy as np
from scipy.stats import norm
import math

from openbox.utils.config_space import Configuration
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.surrogate.base.base_model import AbstractModel
from openbox.surrogate.base.gp import GaussianProcess


class AbstractAcquisitionFunction(object, metaclass=abc.ABCMeta):
    """Abstract base class for acquisition function

    Attributes
    ----------
    model
    logger
    """

    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model: Union[AbstractModel, List[AbstractModel]], **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.
        """
        self.model = model
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the surrogate is updated. E.g.
        entropy search uses it to update its approximation of P(x=x_min),
        EI uses it to update the current optimizer.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, configurations: Union[List[Configuration], np.ndarray], convert=True, **kwargs):
        """Computes the acquisition value for a given X

        Parameters
        ----------
        configurations : list
            The configurations where the acquisition function
            should be evaluated.
        convert : bool

        Returns
        -------
        np.ndarray(N, 1)
            acquisition values for X
        """
        if convert:
            X = convert_configurations_to_array(configurations)
        else:
            X = configurations  # to be compatible with multi-objective acq to call single acq
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X, **kwargs)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError()


class EI(AbstractAcquisitionFunction):
    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the incumbent.
    """

    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(EI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")

        return f


class EIC(EI):
    r"""Computes for a given x the expected constrained improvement as
    acquisition value.

    :math:`\text{EIC}(X) := \text{EI}(X)\prod_{k=1}^K\text{Pr}(c_k(x) \leq 0 | \mathcal{D}_t)`,
    with :math:`c_k \leq 0,\ 1 \leq k \leq K` the constraints, :math:`\mathcal{D}_t` the previous observations.
    """

    def __init__(self,
                 model: AbstractModel,
                 constraint_models: List[GaussianProcess],
                 par: float = 0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(EIC, self).__init__(model, par=par)
        self.constraint_models = constraint_models
        self.long_name = 'Expected Constrained Improvement'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIC value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Constrained Improvement of X
        """
        f = super()._compute(X)
        for model in self.constraint_models:
            m, v = model.predict_marginalized_over_instances(X)
            s = np.sqrt(v)
            f *= norm.cdf(-m / s)
        return f


class EIPS(EI):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        r"""Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) := \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]} {np.log(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X) returning a tuples of
                   predicted cost and running time
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(EIPS, self).__init__(model, par=par)
        self.long_name = 'Expected Improvement per Second'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIPS value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        if m.shape[1] != 2:
            raise ValueError("m has wrong shape: %s != (-1, 2)" % str(m.shape))
        if v.shape[1] != 2:
            raise ValueError("v has wrong shape: %s != (-1, 2)" % str(v.shape))

        m_cost = m[:, 0]
        v_cost = v[:, 0]
        # The surrogate already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m_cost - self.par) / s
            f = (self.eta - m_cost - self.par) * norm.cdf(z) + s * norm.pdf(z)
            f = f / m_runtime
            return f

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()

        if (f < 0).any():
            raise ValueError(
                "Expected Improvement per Second is smaller than 0 "
                "for at least one sample.")

        return f.reshape((-1, 1))


class LogEI(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        r"""Computes for a given x the logarithm expected improvement as
        acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LogEI, self).__init__(model)
        self.long_name = 'Log Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        def calculate_log_ei():
            # we expect that f_min is in log-space
            f_min = self.eta - self.par
            v = (f_min - m) / std
            return (np.exp(f_min) * norm.cdf(v)) - \
                   (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

        if np.any(std == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            std_copy = np.copy(std)
            std[std_copy == 0.0] = 1.0
            log_ei = calculate_log_ei()
            log_ei[std_copy == 0.0] = 0.0
        else:
            log_ei = calculate_log_ei()

        if (log_ei < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample.")

        return log_ei.reshape((-1, 1))


class LPEI(EI):
    def __init__(self,
                 model: AbstractModel,
                 batch_configs=None,
                 par: float = 0.0,
                 estimate_L: float = 10.0,
                 **kwargs):
        r"""This is EI with local penalizer, BBO_LP. Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) := \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]} {np.log(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X) returning a tuples of
                   predicted cost and running time
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LPEI, self).__init__(model, par=par)
        self.estimate_L = estimate_L
        if batch_configs is None:
            batch_configs = []
        self.batch_configs = batch_configs
        self.long_name = 'Expected Improvement with Local Penalizer'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the PenalizedEI value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement per Second of X in log space
        """
        f = super()._compute(X)  # N*1
        f = np.log(np.log1p(np.exp(f)))  # apply g(z), soft-plus transformation
        for config in self.batch_configs:
            m, v = self.model.predict(config.get_array().reshape(1, -1))
            mu = m[0][0]
            var = v[0][0]
            sigma = math.sqrt(var)
            local_penalizer = np.apply_along_axis(self._local_penalizer, 1, X, config.get_array(),
                                                  mu, sigma, self.eta).reshape(-1, 1)
            f += local_penalizer
        return f

    def _local_penalizer(self, x, x_j, mu, sigma, Min):
        L = 5
        # L = self.estimate_L
        r_j = (mu - Min) / L
        s_j = sigma / L
        return norm.logcdf((np.linalg.norm(x - x_j) - r_j) / s_j)


class PI(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):

        """Computes the probability of improvement for a given x over the best so far value as
        acquisition value.

        :math:`P(f_{t+1}(\mathbf{X})\geq f(\mathbf{X^+})) :=
        \Phi(\frac{\mu(\mathbf{X}) - f(\mathbf{X^+})}{\sigma(\mathbf{X})})`,
        with :math:`f(X^+)` as the incumbent and :math:`\Phi` the cdf of the standard normal

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(PI, self).__init__(model)
        self.long_name = 'Probability of Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the PI value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate PI. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<float>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        return norm.cdf((self.eta - m - self.par) / std)


class LCB(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 1.0,
                 **kwargs):

        """Computes the lower confidence bound for a given x over the best so far value as
        acquisition value.

        :math:`LCB(X) = \mu(\mathbf{X}) - \sqrt(\beta_t)\sigma(\mathbf{X})`

        Returns -LCB(X) as the acquisition_function acq_maximizer maximizes the acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LCB, self).__init__(model)
        self.long_name = 'Lower Confidence Bound'
        self.par = par
        self.eta = None  # to be compatible with the existing update calls in SMBO
        self.num_data = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the LCB value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate LCB. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N, 1)
            (Negative) Lower Confidence Bound of X
        """
        if self.num_data is None:
            raise ValueError('No current number of Datapoints specified. Call update('
                             'num_data=<int>) to inform the acquisition function '
                             'about the number of datapoints.')
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        beta = 2 * np.log((X.shape[1] * self.num_data ** 2) / self.par)
        return -(m - np.sqrt(beta) * std)


class Uncertainty(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 1.0,
                 **kwargs):

        """Computes half of the difference between upper and lower confidence bound (Uncertainty).

        :math:`LCB(X) = \mu(\mathbf{X}) - \sqrt(\beta_t)\sigma(\mathbf{X})`
        :math:`UCB(X) = \mu(\mathbf{X}) + \sqrt(\beta_t)\sigma(\mathbf{X})`
        :math:`Uncertainty(X) = \sqrt(\beta_t)\sigma(\mathbf{X})`

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=1.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(Uncertainty, self).__init__(model)
        self.long_name = 'Uncertainty'
        self.par = par
        self.eta = None  # to be compatible with the existing update calls in SMBO
        self.num_data = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the Uncertainty value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate Uncertainty. N is the number of points and D the dimension for the points

        Returns
        -------
        tuple(np.ndarray(N, 1), np.ndarray(N, 1))
            Uncertainty of X
        """
        if self.num_data is None:
            raise ValueError('No current number of Datapoints specified. Call update('
                             'num_data=<int>) to inform the acquisition function '
                             'about the number of datapoints.')
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        beta = 2 * np.log((X.shape[1] * self.num_data ** 2) / self.par)
        uncertainty = np.sqrt(beta) * std
        if np.any(np.isnan(uncertainty)):
            self.logger.warning('Uncertainty has nan-value. Set to 0.')
            uncertainty[np.isnan(uncertainty)] = 0
        return uncertainty
