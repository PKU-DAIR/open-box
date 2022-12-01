# License: MIT

import abc
import numpy as np

from openbox.utils.util_funcs import get_rng
from openbox.utils.logging_utils import get_logger
from openbox.utils.history_container import HistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.core.base import build_acq_func, build_optimizer, build_surrogate


class Advisor(object, metaclass=abc.ABCMeta):
    def __init__(self, config_space,
                 initial_trials=3,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data=None,
                 optimization_strategy='bo',
                 surrogate_type='gp',
                 output_dir='logs',
                 task_id='default_task_id',
                 rng=None):

        # Create output (logging) directory.
        # Init logging module.
        # Random seed generator.
        self.init_strategy = init_strategy
        self.output_dir = output_dir
        if rng is None:
            run_id, rng = get_rng()
        self.rng = rng
        self.logger = get_logger(self.__class__.__name__)

        # Basic components in Advisor.
        self.optimization_strategy = optimization_strategy
        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()
        self.scale_perc = 5
        self.perc = None
        self.min_y = None
        self.max_y = None

        # Init the basic ingredients in Bayesian optimization.
        self.history_bo_data = history_bo_data
        self.surrogate_type = surrogate_type
        self.init_num = initial_trials
        self.config_space = config_space
        self.config_space.seed(rng.randint(MAXINT))

        if initial_configurations is not None and len(initial_configurations) > 0:
            self.initial_configurations = initial_configurations
            self.init_num = len(initial_configurations)
        else:
            self.initial_configurations = self.create_initial_design(self.init_strategy)
            self.init_num = len(self.initial_configurations)
        self.history_container = HistoryContainer(task_id)

        self.surrogate_model = None
        self.acquisition_function = None
        self.optimizer = None
        self.setup_bo_basics()

    def setup_bo_basics(self, acq_type='ei', acq_optimizer_type='local_random'):
        self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                               config_space=self.config_space,
                                               rng=self.rng,
                                               history_hpo_data=self.history_bo_data)

        self.acquisition_function = build_acq_func(func_str=acq_type, model=self.surrogate_model)

        self.optimizer = build_optimizer(func_str=acq_optimizer_type,
                                         acq_func=self.acquisition_function,
                                         config_space=self.config_space,
                                         rng=self.rng)

    def create_initial_design(self, init_strategy='random'):
        default_config = self.config_space.get_default_configuration()
        if init_strategy == 'random':
            num_random_config = self.init_num - 1
            initial_configs = [default_config] + self.sample_random_configs(num_random_config)
            return initial_configs
        elif init_strategy == 'random_explore_first':
            num_random_config = self.init_num - 1
            candidate_configs = self.sample_random_configs(100)
            return self.max_min_distance(default_config,candidate_configs,num_random_config)
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

    def max_min_distance(self, default_config, src_configs, num):
        min_dis = list()
        initial_configs = list()
        initial_configs.append(default_config)

        for config in src_configs:
            dis = np.linalg.norm(config.get_array()-default_config.get_array())  # get_array may have NaN problems
            min_dis.append(dis)
        min_dis = np.array(min_dis)

        for i in range(num):
            furthest_config = src_configs[np.argmax(min_dis)]
            initial_configs.append(furthest_config)
            min_dis[np.argmax(min_dis)] = -1

            for j in range(len(src_configs)):
                if src_configs[j] in initial_configs:
                    continue
                updated_dis = np.linalg.norm(src_configs[j].get_array()-furthest_config.get_array())
                min_dis[j] = min(updated_dis, min_dis[j])

        return initial_configs

    def get_suggestion(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        num_failed_trial = len(self.failed_configurations)
        failed_perfs = list() if self.max_y is None else [self.max_y] * num_failed_trial
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        num_config_evaluated = len(self.perfs + self.failed_configurations)
        if num_config_evaluated < self.init_num:
            return self.initial_configurations[num_config_evaluated]

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(1)[0]
        elif self.optimization_strategy == 'bo':
            self.surrogate_model.train(X, Y)
            incumbent_value = self.history_container.get_incumbents()[0][1]
            self.acquisition_function.update(model=self.surrogate_model,
                                             eta=incumbent_value,
                                             num_data=num_config_evaluated)
            challengers = self.optimizer.maximize(runhistory=self.history_container,
                                                  num_points=5000)
            is_repeated_config = True
            repeated_time = 0
            cur_config = None
            while is_repeated_config:
                cur_config = challengers.challengers[repeated_time]
                if cur_config in (self.configurations + self.failed_configurations):
                    repeated_time += 1
                else:
                    is_repeated_config = False
            return cur_config
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

    def update_observation(self, observation):
        config, perf, trial_state = observation
        if trial_state == SUCCESS and perf < MAXINT:
            self.configurations.append(config)
            self.perfs.append(perf)
            self.history_container.add(config, perf)

            self.perc = np.percentile(self.perfs, self.scale_perc)
            self.min_y = np.min(self.perfs)
            self.max_y = np.max(self.perfs)
        else:
            self.failed_configurations.append(config)

    def sample_random_configs(self, num_configs=1):
        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in (self.configurations + self.failed_configurations + configs):
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs

    def get_suggestions(self):
        raise NotImplementedError
