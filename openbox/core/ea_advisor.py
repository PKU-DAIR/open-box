# License: MIT

import abc
import random
import numpy as np

from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.config_space import get_one_exchange_neighbourhood


class EA_Advisor(object, metaclass=abc.ABCMeta):
    """
    Evolutionary Algorithm Advisor
    """

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            population_size=30,
            subset_size=20,
            epsilon=0.2,
            strategy='worst',  # 'worst', 'oldest'
            optimization_strategy='ea',
            batch_size=1,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
    ):

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        assert self.num_objectives == 1 and self.num_constraints == 0
        self.output_dir = output_dir
        self.task_id = task_id
        self.rng = check_random_state(random_state)
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)

        _logger_kwargs = {'name': task_id, 'logdir': output_dir}
        _logger_kwargs.update(logger_kwargs or {})
        logger.init(**_logger_kwargs)

        # Init parallel settings
        self.batch_size = batch_size
        self.init_num = batch_size  # for compatibility in pSMBO
        self.running_configs = list()

        # Basic components in Advisor.
        self.optimization_strategy = optimization_strategy

        # Init the basic ingredients
        self.all_configs = set()
        self.age = 0
        self.population = list()
        self.population_size = population_size
        self.subset_size = subset_size
        assert 0 < self.subset_size <= self.population_size
        self.epsilon = epsilon
        self.strategy = strategy
        assert self.strategy in ['worst', 'oldest']

        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints, config_space=config_space,
            ref_point=None, meta_info=None,  # todo: add meta info
        )

    def get_suggestion(self, history=None):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history is None:
            history = self.history

        if len(self.population) < self.population_size:
            # Initialize population
            next_config = self.sample_random_config(excluded_configs=self.all_configs)
        else:
            # Select a parent by subset tournament and epsilon greedy
            if self.rng.random() < self.epsilon:
                parent_config = random.sample(self.population, 1)[0]['config']
            else:
                subset = random.sample(self.population, self.subset_size)
                subset.sort(key=lambda x: x['perf'])    # minimize
                parent_config = subset[0]['config']

            # Mutation to 1-step neighbors
            next_config = None
            neighbors_gen = get_one_exchange_neighbourhood(parent_config, seed=self.rng.randint(MAXINT))
            for neighbor in neighbors_gen:
                if neighbor not in self.all_configs:
                    next_config = neighbor
                    break
            if next_config is None:  # If all the neighors are evaluated, sample randomly!
                next_config = self.sample_random_config(excluded_configs=self.all_configs)

        self.all_configs.add(next_config)
        self.running_configs.append(next_config)
        return next_config

    def get_suggestions(self, batch_size=None, history=None):
        if batch_size is None:
            batch_size = self.batch_size

        configs = list()
        for i in range(batch_size):
            config = self.get_suggestion(history)
            configs.append(config)
        return configs

    def update_observation(self, observation: Observation):
        """
        Update the current observations.
        Parameters
        ----------
        observation

        Returns
        -------

        """

        config = observation.config
        perf = observation.objectives[0]
        trial_state = observation.trial_state

        assert config in self.running_configs
        self.running_configs.remove(config)

        # update population
        if trial_state == SUCCESS and perf < np.inf:
            self.population.append(dict(config=config, age=self.age, perf=perf))
            self.age += 1

        # Eliminate samples
        if len(self.population) > self.population_size:
            if self.strategy == 'oldest':
                self.population.sort(key=lambda x: x['age'])
                self.population.pop(0)
            elif self.strategy == 'worst':
                self.population.sort(key=lambda x: x['perf'])
                self.population.pop(-1)
            else:
                raise ValueError('Unknown strategy: %s' % self.strategy)

        return self.history.update_observation(observation)

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
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                break
        return config

    def get_history(self):
        return self.history
