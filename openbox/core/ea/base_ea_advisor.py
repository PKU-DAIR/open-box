
import abc
import numpy as np
import random

from typing import *

from openbox.utils.util_funcs import check_random_state
from openbox.utils.logging_utils import get_logger
from openbox.utils.history_container import HistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.config_space import get_one_exchange_neighbourhood
from openbox.core.base import Observation

from ConfigSpace import ConfigurationSpace

class EAAdvisor:


    def __init__(self, config_space: ConfigurationSpace,
                 num_objs=1,
                 num_constraints=0,
                 population_size=30,
                 optimization_strategy='ea',
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None):

        # System Settings.

        self.rng = check_random_state(random_state)
        self.output_dir = output_dir
        self.logger = get_logger(self.__class__.__name__)

        # Objectives Settings
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)

        # TODO support multiple objects and constraints.
        assert self.num_objs == 1 and self.num_constraints == 0



        # Init parallel settings
        self.batch_size = batch_size
        self.init_num = batch_size  # for compatibility in pSMBO
        self.running_configs = list()

        # Basic components in Advisor.
        # The parameter should be removed. Keep it here to avoid compatibility issues.
        self.optimization_strategy = optimization_strategy

        # Start initialization for EA variables.
        self.all_configs = set()
        self.age = 0
        self.population = list()
        self.population_size = population_size

        # init history container
        self.history_container = HistoryContainer(task_id, self.num_constraints, config_space=self.config_space)

    def get_suggestion(self):
        # TODO should there be a history_container param here?
        raise NotImplementedError

    def update_observation(self, observation: Observation):
        raise NotImplementedError

    def get_suggestions(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        return [self.get_suggestion() for i in range(batch_size)]

    def update_observations(self, observations: List[Observation]):
        return [self.update_observation(o) for o in observations]

    def sample_random_config(self, excluded_configs=None):
        if excluded_configs is None:
            excluded_configs = set()

        sample_cnt = 0
        max_sample_cnt = 1000
        while True:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in excluded_configs:
                break
            if sample_cnt >= max_sample_cnt:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                break
        return config

    def get_history(self):
        return self.history_container
