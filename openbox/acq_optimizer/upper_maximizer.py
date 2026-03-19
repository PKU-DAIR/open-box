import abc
import time
import warnings
from typing import Iterable, List, Union, Tuple, Optional,Any
import random
import scipy.optimize
import numpy as np

from openbox import logger
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.utils.config_space import get_one_exchange_neighbourhood, \
    Configuration, ConfigurationSpace
from openbox.utils.history import History, MultiStartHistory
from openbox.utils.util_funcs import get_types
from openbox.utils.constants import MAXINT
from ..compressor.sampling import SamplingStrategy, StandardSamplingStrategy
from . import generator
from . import base
from . import selector

class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rng: Union[bool, np.random.RandomState] = None,
            turbo_length=None,
    ):
        if sampling_strategy is None:
            sampling_strategy = StandardSamplingStrategy(config_space, seed=getattr(rng, "randint", lambda *_: None)(MAXINT))
        self.sampling_strategy = sampling_strategy
        self.config_space = config_space
        self.turbo_length=turbo_length
        
        self.turbo_state=False
        if self.turbo_length is not None:
            self.turbo_state=True
        
        if rng is None:
            logger.debug('no rng given, using default seed of 1')
            self.rng = np.random.RandomState(seed=1)
        else:
            self.rng = rng

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consisting of :class:`openbox.config_space.Configuration`.
        """
        return [t[1] for t in self._maximize(acquisition_function, history, num_points, **kwargs)]

    @abc.abstractmethod
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(
            self,
            acquisition_function,
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]

    def fliter(
            self,
            incumbent_config,
            challengers):
        x_center = incumbent_config.get_array()
        lower_bounds = x_center - self.turbo_length / 2.0
        upper_bounds = x_center + self.turbo_length / 2.0
        filtered_challengers = []
        for config in challengers:
            config_array = config.get_array()
            if np.all(config_array >= lower_bounds) and np.all(config_array <= upper_bounds):
                filtered_challengers.append(config)
        return filtered_challengers


class InterleavedLocalAndRandomSearchMaximizer(AcquisitionFunctionMaximizer):
    """Implements openbox's default acquisition function optimization.

    This acq_optimizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations and interleaves unsorted, randomly
    sampled configurations in between.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        [LocalSearchMaximizer] Maximum number of steps that the local search will perform

    n_steps_plateau_walk: int
        [LocalSearchMaximizer] number of steps during a plateau walk before local search terminates

    n_sls_iterations: int
        [LocalSearchMaximizer] number of local search iterations

    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            sampling_strategy:SamplingStrategy = None,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            n_sls_iterations: int = 10,
            rand_prob=0.25,
            turbo_length=None,
    ):
        super().__init__(config_space, sampling_strategy, rng, turbo_length)
        
        self.random_generator = generator.RandomSearchGenerator(
            config_space=config_space,
            sampling_strategy=self.sampling_strategy,
            random_state=None,
        )
        self.local_generator = generator.LocalSearchGenerator(
            config_space=config_space,
            sampling_strategy=self.sampling_strategy,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
        )
        self.n_sls_iterations = n_sls_iterations
        self.strategy=[self.local_generator,self.random_generator]
        
        # =======================================================================
        # self.local_search = DiffOpt(
        #     acquisition_function=acquisition_function,
        #     config_space=config_space,
        #     rng=rng
        # )
        # =======================================================================

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            List of configurations.
        """
        self.optimizer = base.QuotaCompositeOptimizer(acquisition_function=acquisition_function,
                                                      config_space=self.config_space,
                                                      strategies=self.strategy,
                                                      quotas=[self.n_sls_iterations, num_points - self.n_sls_iterations],
                                                      rng=self.rng,
                                                      candidate_multiplier=3)
        results = self.optimizer._maximize(history=history,
                                           num_points=num_points,
                                           **kwargs)
        
        challengers=[]
        
        for _ in results:
            config=_[1]
            challengers.append(config)
            
        if self.turbo_state and history:
            incumbent_config = self.rng.choice(history.get_incumbent_configs())
            flitered_challengers=self.fliter(incumbent_config=incumbent_config,challengers=challengers)
            return flitered_challengers
            
        return challengers


    def _maximize(
            self,
            history: History,
            acquisition_function:AbstractAcquisitionFunction,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        self.optimizer = base.QuotaCompositeOptimizer(acquisition_function=acquisition_function,
                                                      config_space=self.config_space,
                                                      strategies=self.strategy,
                                                      quotas=[self.n_sls_iterations, num_points - self.n_sls_iterations],
                                                      rng=self.rng,
                                                      candidate_multiplier=3)
        results = self.optimizer._maximize(history=history,
                                           num_points=num_points,
                                           **kwargs)

        return results