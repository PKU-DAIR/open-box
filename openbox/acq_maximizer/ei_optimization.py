# License: MIT
# This file is partially built on SMAC3(https://github.com/automl/SMAC3), which is licensed as follows,

# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)
# Author: Aaron Klein, Marius Lindauer

import abc
import time
import warnings
from typing import Iterable, List, Union, Tuple, Optional
import random
import scipy.optimize
import numpy as np

from openbox import logger
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.utils.config_space import get_one_exchange_neighbourhood, \
    Configuration, ConfigurationSpace
from openbox.acq_maximizer.random_configuration_chooser import ChooserNoCoolDown, ChooserProb
from openbox.utils.history import History, MultiStartHistory
from openbox.utils.util_funcs import get_types
from openbox.utils.constants import MAXINT


class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None
    ):
        self.acquisition_function = acquisition_function
        self.config_space = config_space

        if rng is None:
            logger.debug('no rng given, using default seed of 1')
            self.rng = np.random.RandomState(seed=1)
        else:
            self.rng = rng

    def maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consisting of :class:`openbox.config_space.Configuration`.
        """
        return [t[1] for t in self._maximize(runhistory, num_points, **kwargs)]

    @abc.abstractmethod
    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
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
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = self.acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]


class CMAESOptimizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            rand_prob=0.25,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()

    def maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        try:
            from cma import CMAEvolutionStrategy
        except ImportError:
            raise ImportError("Package cma is not installed!")

        types, bounds = get_types(self.config_space)
        assert all(types == 0)

        # Check Constant Hyperparameter
        const_idx = list()
        for i, bound in enumerate(bounds):
            if np.isnan(bound[1]):
                const_idx.append(i)

        hp_num = len(bounds) - len(const_idx)
        es = CMAEvolutionStrategy(hp_num * [0], 0.99, inopts={'bounds': [0, 1]})

        eval_num = 0
        next_configs_by_acq_value = list()
        while eval_num < num_points:
            X = es.ask(number=es.popsize)
            _X = X.copy()
            for i in range(len(_X)):
                for index in const_idx:
                    _X[i] = np.insert(_X[i], index, 0)
            _X = np.asarray(_X)
            values = self.acquisition_function._compute(_X)
            values = np.reshape(values, (-1,))
            es.tell(X, values)
            next_configs_by_acq_value.extend([(values[i], _X[i]) for i in range(es.popsize)])
            eval_num += es.popsize

        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]
        next_configs_by_acq_value = [Configuration(self.config_space, vector=array) for array in
                                     next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value,
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers


class LocalSearch(AcquisitionFunctionMaximizer):
    """Implementation of openbox's local search.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        Maximum number of iterations that the local search will perform

    n_steps_plateau_walk: int
        number of steps during a plateau walk before local search terminates

    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
        num_points: int
            number of points to be sampled
        **kwargs:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found configuration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """

        init_points = self._get_initial_points(
            num_points, runhistory)

        acq_configs = []
        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, configuration = self._one_iter(
                start_point, **kwargs)

            configuration.origin = "Local Search"
            acq_configs.append((acq_val, configuration))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        return acq_configs

    def _get_initial_points(self, num_points, runhistory):

        if runhistory.empty():
            init_points = self.config_space.sample_configuration(
                size=num_points)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = runhistory.configurations
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_points)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )

        return init_points

    def _one_iter(
            self,
            start_point: Configuration,
            **kwargs
    ) -> Tuple[float, Configuration]:

        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.acquisition_function([incumbent], **kwargs)[0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                logger.warning(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps
                )

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(
                incumbent, seed=self.rng.randint(MAXINT))

            for neighbor in all_neighbors:
                s_time = time.time()
                acq_val = self.acquisition_function([neighbor], **kwargs)
                neighbors_looked_at += 1
                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent:
                    # logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or \
                    (self.max_steps is not None and
                     local_search_steps == self.max_steps):
                if len(time_n) == 0:
                    time_n.append(0.0)
                logger.debug("Local search took %d steps and looked at %d "
                                  "configurations. Computing the acquisition "
                                  "value for one configuration took %f seconds"
                                  " on average.",
                                  local_search_steps, neighbors_looked_at,
                                  np.mean(time_n))
                break

        return acq_val_incumbent, incumbent


class RandomSearch(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            _sorted: bool = False,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Randomly sampled configurations

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            not used

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """

        if num_points > 1:
            rand_configs = self.config_space.sample_configuration(
                size=num_points)
        else:
            rand_configs = [self.config_space.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (sorted)'
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]


class InterleavedLocalAndRandomSearch(AcquisitionFunctionMaximizer):
    """Implements openbox's default acquisition function optimization.

    This acq_maximizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations and interleaves unsorted, randomly
    sampled configurations in between.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        [LocalSearch] Maximum number of steps that the local search will perform

    n_steps_plateau_walk: int
        [LocalSearch] number of steps during a plateau walk before local search terminates

    n_sls_iterations: int
        [Local Search] number of local search iterations

    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            n_sls_iterations: int = 10,
            rand_prob=0.25
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_search = RandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng
        )
        self.local_search = LocalSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk
        )
        self.n_sls_iterations = n_sls_iterations
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)

        # =======================================================================
        # self.local_search = DiffOpt(
        #     acquisition_function=acquisition_function,
        #     config_space=config_space,
        #     rng=rng
        # )
        # =======================================================================

    def maximize(
            self,
            runhistory: History,
            num_points: int,
            random_configuration_chooser=None,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
        num_points: int
            number of points to be sampled
        random_configuration_chooser: ~openbox.acq_maximizer.random_configuration_chooser.RandomConfigurationChooser
            part of the returned ChallengerList such
            that we can interleave random configurations
            by a scheme defined by the random_configuration_chooser;
            random_configuration_chooser.next_smbo_iteration()
            is called at the end of this function
        **kwargs
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            to be concrete: ~openbox.ei_optimization.ChallengerList
        """

        next_configs_by_local_search = self.local_search._maximize(
            runhistory, self.n_sls_iterations, **kwargs
        )

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            runhistory,
            num_points - len(next_configs_by_local_search),
            _sorted=True,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of openbox. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (
                next_configs_by_random_search_sorted
                + next_configs_by_local_search
        )
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s",
            str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])
        )
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value,
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class ScipyOptimizer(AcquisitionFunctionMaximizer):
    """
    Wraps scipy optimizer. Only on continuous dims.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)

        types, bounds = get_types(self.config_space)    # todo: support constant hp in scipy optimizer
        assert all(types == 0), 'Scipy optimizer (L-BFGS-B) only supports Integer and Float parameters.'
        self.bounds = bounds

        options = dict(disp=False, maxiter=1000)
        self.scipy_config = dict(tol=None, method='L-BFGS-B', options=options)

    def maximize(
            self,
            runhistory: History,
            initial_config=None,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:

        def negative_acquisition(x):
            # shape of x = (d,)
            x = np.clip(x, 0.0, 1.0)    # fix numerical problem in L-BFGS-B
            try:
                # self.config_space._check_forbidden(x)
                Configuration(self.config_space, vector=x).is_valid_configuration()
            except ValueError:
                return np.inf
            return -self.acquisition_function(x, convert=False)[0]  # shape=(1,)

        if initial_config is None:
            initial_config = self.config_space.sample_configuration()
        init_point = initial_config.get_array()

        acq_configs = []
        with warnings.catch_warnings():
            # ignore warnings of np.inf
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
            result = scipy.optimize.minimize(fun=negative_acquisition,
                                             x0=init_point,
                                             bounds=self.bounds,
                                             **self.scipy_config)
        if not result.success:
            logger.debug('Scipy optimizer failed. Info:\n%s' % (result,))
        try:
            x = np.clip(result.x, 0.0, 1.0)  # fix numerical problem in L-BFGS-B
            config = Configuration(self.config_space, vector=x)
            config.is_valid_configuration()
            acq = self.acquisition_function(x, convert=False)
            acq_configs.append((acq, config))
        except Exception:
            pass

        if not acq_configs:  # empty
            logger.warning('Scipy optimizer failed. Return empty config list. Info:\n%s' % (result,))

        challengers = ChallengerList([config for _, config in acq_configs],
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class RandomScipyOptimizer(AcquisitionFunctionMaximizer):
    """
    Use scipy.optimize with start points chosen by random search. Only on continuous dims.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)

        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)

        self.random_search = InterleavedLocalAndRandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng
        )
        self.scipy_optimizer = ScipyOptimizer(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng
        )

    def maximize(
            self,
            runhistory: History,
            num_points: int,
            num_trials=10,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        assert num_trials >= 3
        acq_configs = []

        initial_configs = self.random_search.maximize(runhistory, num_points, **kwargs).challengers
        initial_acqs = self.acquisition_function(initial_configs)
        acq_configs.extend(zip(initial_acqs, initial_configs))

        scipy_initial_configs = [initial_configs[0]] + self.config_space.sample_configuration(num_trials - 1)
        success_count = 0
        for config in scipy_initial_configs:
            scipy_configs = self.scipy_optimizer.maximize(runhistory, initial_config=config).challengers
            if not scipy_configs:   # empty
                continue
            scipy_acqs = self.acquisition_function(scipy_configs)
            acq_configs.extend(zip(scipy_acqs, scipy_configs))
            success_count += 1
        if success_count == 0:
            logger.warning('None of Scipy optimizations are successful in RandomScipyOptimizer.')

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        configs = [_[1] for _ in acq_configs]

        challengers = ChallengerList(configs,
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class ScipyGlobalOptimizer(AcquisitionFunctionMaximizer):
    """
    Wraps scipy global optimizer. Only on continuous dims.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)

        types, bounds = get_types(self.config_space)
        assert all(types == 0)
        self.bounds = bounds

    def maximize(
            self,
            runhistory: History,
            initial_config=None,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:

        def negative_acquisition(x):
            # shape of x = (d,)
            return -self.acquisition_function(x, convert=False)[0]  # shape=(1,)

        acq_configs = []
        result = scipy.optimize.differential_evolution(func=negative_acquisition,
                                                       bounds=self.bounds)
        if not result.success:
            logger.debug('Scipy differential evolution optimizer failed. Info:\n%s' % (result,))
        try:
            config = Configuration(self.config_space, vector=result.x)
            acq = self.acquisition_function(result.x, convert=False)
            acq_configs.append((acq, config))
        except Exception:
            pass

        if not acq_configs:  # empty
            logger.warning('Scipy differential evolution optimizer failed. Return empty config list. Info:\n%s' % (result,))

        challengers = ChallengerList([config for _, config in acq_configs],
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class StagedBatchScipyOptimizer(AcquisitionFunctionMaximizer):
    """ todo constraints
    Use batch scipy.optimize with start points chosen by specific method. Only on continuous dims.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    num_random : Number of random chosen points

    num_restarts : The number of starting points for multistart acquisition
            function optimization

    raw_samples : The number of samples for initialization

    batch_limit : Number of points in a batch optimized jointly by scipy minimizer

    scipy_maxiter : Maximum number of scipy minimizer iterations to perform

    rand_prob : Probability of choosing random config

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            num_random: int = 1000,
            num_restarts: int = 20,
            raw_samples: int = 1024,
            batch_limit: int = 5,
            scipy_maxiter: int = 200,
            rand_prob: float = 0.0,
            rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.num_random = num_random
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.batch_limit = batch_limit
        self.scipy_max_iter = scipy_maxiter
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)
        self.minimizer = scipy.optimize.minimize
        self.method = "L-BFGS-B"
        self.dim = len(self.config_space.get_hyperparameters())
        self.bound = (0.0, 1.0)  # todo only on continuous dims (int, float) now

    def gen_initial_points(self, num_restarts, raw_samples):
        # todo other strategy
        random_points = self.rng.uniform(self.bound[0], self.bound[1], size=(raw_samples, self.dim))
        acq_random = self.acquisition_function(random_points, convert=False).reshape(-1)
        idx = np.argsort(acq_random)[::-1][:num_restarts]
        return random_points[idx]

    def gen_batch_scipy_points(self, initial_points: np.ndarray):
        #count = 0  # todo remove
        def f(X_flattened):
            # nonlocal count
            # count += 1
            X = X_flattened.reshape(shapeX)
            joint_acq = -self.acquisition_function(X, convert=False).sum().item()
            return joint_acq

        shapeX = initial_points.shape
        x0 = initial_points.reshape(-1)
        bounds = [self.bound] * x0.shape[0]

        result = self.minimizer(
            f,
            x0=x0,
            method=self.method,
            bounds=bounds,
            options=dict(maxiter=self.scipy_max_iter),
        )
        #print('count=', count)  # todo remove

        # return result.x even failed. may because 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
        # if not result.success:
        #     logger.warning('Scipy minimizer %s failed in this round: %s.' % (self.method, result))
        #     return None

        #print(result.x.reshape(shapeX))    # todo remove
        return result.x.reshape(shapeX)

    def maximize(
            self,
            runhistory: History,
            num_points: int,  # todo useless
            **kwargs
    ) -> List[Tuple[float, Configuration]]:

        # print('start optimize')   # todo remove
        # import time
        # t0 = time.time()
        acq_configs = []

        # random points
        random_points = self.rng.uniform(self.bound[0], self.bound[1], size=(self.num_random, self.dim))
        acq_random = self.acquisition_function(random_points, convert=False)
        for i in range(random_points.shape[0]):
            # convert array to Configuration
            config = Configuration(self.config_space, vector=random_points[i])
            config.origin = 'Random Search'
            acq_configs.append((acq_random[i], config))

        # scipy points
        initial_points = self.gen_initial_points(num_restarts=self.num_restarts, raw_samples=self.raw_samples)

        for start_idx in range(0, self.num_restarts, self.batch_limit):
            end_idx = min(start_idx + self.batch_limit, self.num_restarts)
            # optimize using random restart optimization
            scipy_points = self.gen_batch_scipy_points(initial_points[start_idx:end_idx])
            if scipy_points is None:
                continue
            acq_scipy = self.acquisition_function(scipy_points, convert=False)
            for i in range(scipy_points.shape[0]):
                # convert array to Configuration
                config = Configuration(self.config_space, vector=scipy_points[i])
                config.origin = 'Batch Scipy'
                acq_configs.append((acq_scipy[i], config))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        configs = [_[1] for _ in acq_configs]

        challengers = ChallengerList(configs,
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()

        # t1 = time.time()  # todo remove
        # print('==time total=%.2f' % (t1-t0,))
        # for x1 in np.linspace(0, 1, 20):
        #     optimal_point = np.array([x1.item()] + [0.5] * (self.dim-1))
        #     print('optimal_point acq=', self.acquisition_function(optimal_point, convert=False))
        # print('best point acq=', acq_configs[0])
        # time.sleep(2)
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class MESMO_Optimizer(AcquisitionFunctionMaximizer):
    """Implements Scipy optimizer for MESMO. Only on continuous dims

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            num_mc=1000,
            num_opt=1000,
            rand_prob=0.0
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)
        self.num_mc = num_mc
        self.num_opt = num_opt
        self.minimizer = scipy.optimize.minimize

    def maximize(
            self,
            runhistory: History,
            num_points: int,  # todo useless
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
        num_points: int
            number of points to be sampled
        **kwargs
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            to be concrete: ~openbox.ei_optimization.ChallengerList
        """

        def inverse_acquisition(x):
            # shape of x = (d,)
            return -self.acquisition_function(x, convert=False)[0]  # shape=(1,)

        d = len(self.config_space.get_hyperparameters())
        bound = (0.0, 1.0)  # todo only on continuous dims (int, float) now
        bounds = [bound] * d
        acq_configs = []

        # MC
        x_tries = self.rng.uniform(bound[0], bound[1], size=(self.num_mc, d))
        acq_tries = self.acquisition_function(x_tries, convert=False)
        for i in range(x_tries.shape[0]):
            # convert array to Configuration
            config = Configuration(self.config_space, vector=x_tries[i])
            config.origin = 'Random Search'
            acq_configs.append((acq_tries[i], config))

        # L-BFGS-B
        x_seed = self.rng.uniform(low=bound[0], high=bound[1], size=(self.num_opt, d))
        for i in range(x_seed.shape[0]):
            x0 = x_seed[i].reshape(1, -1)
            result = self.minimizer(inverse_acquisition, x0=x0, method='L-BFGS-B', bounds=bounds)
            if not result.success:
                continue
            # convert array to Configuration
            config = Configuration(self.config_space, vector=result.x)
            config.origin = 'Scipy'
            acq_val = self.acquisition_function(result.x, convert=False)  # [0]
            acq_configs.append((acq_val, config))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        configs = [_[1] for _ in acq_configs]

        challengers = ChallengerList(configs,
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class USeMO_Optimizer(AcquisitionFunctionMaximizer):
    """Implements USeMO optimizer

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            rand_prob=0.0
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)

    def maximize(
            self,
            runhistory: History,
            num_points: int,  # useless in USeMO
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
        num_points: int
            number of points to be sampled
        **kwargs
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            to be concrete: ~openbox.ei_optimization.ChallengerList
        """

        acq_vals = np.asarray(self.acquisition_function.uncertainties)
        candidates = np.asarray(self.acquisition_function.candidates)
        assert len(acq_vals.shape) == 1 and len(candidates.shape) == 2 \
               and acq_vals.shape[0] == candidates.shape[0]

        acq_configs = []
        for i in range(acq_vals.shape[0]):
            # convert array to Configuration todo
            config = Configuration(self.config_space, vector=candidates[i])
            acq_configs.append((acq_vals[i], config))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        configs = [_[1] for _ in acq_configs]

        challengers = ChallengerList(configs,
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class batchMCOptimizer(AcquisitionFunctionMaximizer):
    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            batch_size=None,
            rand_prob=0.0
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)

        if batch_size is None:
            types, bounds = get_types(self.config_space)
            dim = np.sum(types == 0)
            self.batch_size = min(5000, max(2000, 200 * dim))
        else:
            self.batch_size = batch_size

    def maximize(
            self,
            runhistory: Union[History, MultiStartHistory],
            num_points: int,
            _sorted: bool = True,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Randomly sampled configurations

        Parameters
        ----------
        runhistory: openbox.utils.history.History
            runhistory object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            turbo_state: TurboState
                provide turbo state to use trust region

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """
        from openbox.utils.samplers import SobolSampler

        cur_idx = 0
        config_acq = list()
        weight_seed = self.rng.randint(0, int(1e8))  # The same weight seed each iteration

        while cur_idx < num_points:
            batch_size = min(self.batch_size, num_points - cur_idx)
            turbo_state = kwargs.get('turbo_state', None)
            if turbo_state is None:
                lower_bounds = None
                upper_bounds = None
            else:
                assert isinstance(runhistory, MultiStartHistory)
                if runhistory.num_objectives > 1:
                    # TODO implement adaptive strategy to choose trust region center for MO
                    raise NotImplementedError()
                else:
                    incumbent_config = self.rng.choice(runhistory.get_incumbent_configs())
                    x_center = incumbent_config.get_array()
                    lower_bounds = x_center - turbo_state.length / 2.0
                    upper_bounds = x_center + turbo_state.length / 2.0

            sobol_sampler = SobolSampler(self.config_space, batch_size,
                                         lower_bounds, upper_bounds,
                                         random_state=self.rng.randint(0, int(1e8)))
            _configs = sobol_sampler.generate(return_config=True)
            _acq_values = self.acquisition_function(_configs, seed=weight_seed)
            config_acq.extend([(_configs[idx], _acq_values[idx]) for idx in range(len(_configs))])

            cur_idx += self.batch_size

        config_acq.sort(reverse=True, key=lambda x: x[1])

        challengers = ChallengerList([_[0] for _ in config_acq],
                                     self.config_space,
                                     self.random_chooser)
        self.random_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class ChallengerList(object):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    challengers : list
        List of challengers (without interleaved random configurations)

    configuration_space : ConfigurationSpace
        ConfigurationSpace from which to sample new random configurations.
    """

    def __init__(self, challengers, configuration_space, random_configuration_chooser=ChooserNoCoolDown(2.0)):
        self.challengers = challengers
        self.configuration_space = configuration_space
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self.random_configuration_chooser = random_configuration_chooser

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.challengers):
            raise StopIteration
        else:
            if self.random_configuration_chooser.check(self._iteration):
                config = self.configuration_space.sample_configuration()
                config.origin = 'Random Search'
            else:
                config = self.challengers[self._index]
                self._index += 1
            self._iteration += 1
            return config
