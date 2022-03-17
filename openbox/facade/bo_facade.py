# License: MIT

import os
import sys
import abc
import traceback
import numpy as np
from openbox.utils.history_container import HistoryContainer
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb
from openbox.utils.util_funcs import get_types, get_rng
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.logging_utils import setup_logger, get_logger


class BaseFacade(object, metaclass=abc.ABCMeta):
    def __init__(self, config_space, task_id, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.logger = None
        self.history_container = HistoryContainer(task_id)
        self.config_space = config_space
        self.scale_perc = 5
        self.perc = None
        self.min_y = None
        self.max_y = None

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def iterate(self):
        raise NotImplementedError()

    def get_history(self):
        return self.history_container

    def get_incumbent(self):
        return self.history_container.get_incumbents()

    def _get_logger(self, name):
        logger_name = 'open-box-%s' % name
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)


class BayesianOptimization(BaseFacade):
    def __init__(self, objective_function, config_space,
                 sample_strategy='bo',
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 initial_runs=3,
                 task_id='default_task_id',
                 rng=None):
        super().__init__(config_space, task_id, output_dir=logging_dir)
        self.logger = super()._get_logger(self.__class__.__name__)
        if rng is None:
            run_id, rng = get_rng()

        self.init_num = initial_runs
        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        self.time_limit_per_trial = time_limit_per_trial
        self.default_obj_value = MAXINT
        self.sample_strategy = sample_strategy

        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()

        # Initialize the basic component in BO.
        self.config_space.seed(rng.randint(MAXINT))
        self.objective_function = objective_function
        types, bounds = get_types(config_space)
        # TODO: what is the feature array.
        self.model = RandomForestWithInstances(types=types, bounds=bounds, seed=rng.randint(MAXINT))
        self.acquisition_function = EI(self.model)
        self.optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.acquisition_function,
            config_space=self.config_space,
            rng=np.random.RandomState(seed=rng.randint(MAXINT)),
            max_steps=self.sls_max_steps,
            n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
            n_sls_iterations=self.n_sls_iterations
        )
        self._random_search = RandomSearch(
            self.acquisition_function, self.config_space, rng
        )
        self.random_configuration_chooser = ChooserProb(prob=0.25, rng=rng)

    def run(self):
        while self.iteration_id < self.max_iterations:
            self.iterate()

    def iterate(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        failed_perfs = list() if self.max_y is None else [self.max_y] * len(self.failed_configurations)
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        config = self.choose_next(X, Y)

        trial_state = SUCCESS
        trial_info = None

        if config not in (self.configurations + self.failed_configurations):
            # Evaluate this configuration.
            try:
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function, self.time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % self.time_limit_per_trial)
                else:
                    perf = MAXINT if _result is None else _result
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                perf = MAXINT
                trial_info = str(e)
                self.logger.error(trial_info)

            if trial_state == SUCCESS and perf < MAXINT:
                if len(self.configurations) == 0:
                    self.default_obj_value = perf

                self.configurations.append(config)
                self.perfs.append(perf)
                self.history_container.add(config, perf)

                self.perc = np.percentile(self.perfs, self.scale_perc)
                self.min_y = np.min(self.perfs)
                self.max_y = np.max(self.perfs)
            else:
                self.failed_configurations.append(config)
        else:
            self.logger.debug('This configuration has been evaluated! Skip it.')
            if config in self.configurations:
                config_idx = self.configurations.index(config)
                trial_state, perf = SUCCESS, self.perfs[config_idx]
            else:
                trial_state, perf = FAILED, MAXINT

        self.iteration_id += 1
        self.logger.info(
            'Iteration-%d, objective improvement: %.4f' % (self.iteration_id, max(0, self.default_obj_value - perf)))
        return config, trial_state, perf, trial_info

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        _config_num = X.shape[0]
        if _config_num < self.init_num:
            default_config = self.config_space.get_default_configuration()
            if default_config not in (self.configurations + self.failed_configurations):
                return default_config
            else:
                return self._random_search.maximize(runhistory=self.history_container, num_points=1)[0]

        if self.sample_strategy == 'random':
            return self.sample_config()
        elif self.sample_strategy == 'bo':
            if self.random_configuration_chooser.check(self.iteration_id):
                return self.sample_config()
            else:
                self.model.train(X, Y)

                incumbent_value = self.history_container.get_incumbents()[0][1]

                self.acquisition_function.update(model=self.model, eta=incumbent_value,
                                                 num_data=len(self.history_container.data))

                challengers = self.optimizer.maximize(
                    runhistory=self.history_container,
                    num_points=5000,
                    random_configuration_chooser=self.random_configuration_chooser
                )

                return challengers.challengers[0]
        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.sample_strategy)

    def sample_config(self):
        config = None
        _sample_cnt, _sample_limit = 0, 10000
        while True:
            _sample_cnt += 1
            config = self.config_space.sample_configuration()
            if config not in (self.configurations + self.failed_configurations):
                break
            if _sample_cnt >= _sample_limit:
                config = self.config_space.sample_configuration()
                break
        return config
