# License: MIT

import abc
import numpy as np
import random

from openbox.core.ea.base_ea_advisor import EAAdvisor
from openbox.utils.util_funcs import check_random_state
from openbox.utils.logging_utils import get_logger
from openbox.utils.history_container import HistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.config_space import get_one_exchange_neighbourhood
from openbox.core.base import Observation

from openbox.core.ea.base_ea_advisor import Individual, constraint_check


class RegularizedEAAdvisor(EAAdvisor, metaclass=abc.ABCMeta):
    """
    Evolutionary Algorithm Advisor
    """

    def __init__(self,

                 config_space,
                 num_objs=1,
                 num_constraints=0,
                 population_size=30,
                 optimization_strategy='ea',
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None,

                 constraint_strategy='discard',

                 subset_size=20,
                 epsilon=0.2,
                 strategy='worst',
                 ):


        EAAdvisor.__init__(self, config_space, num_objs=num_objs, num_constraints=num_constraints,
                           population_size=population_size, optimization_strategy=optimization_strategy,
                           batch_size=batch_size, output_dir=output_dir, task_id=task_id, random_state=random_state,
                           )

        assert num_objs == 1
        assert constraint_strategy == 'discard'
        self.constraint_strategy = constraint_strategy

        self.subset_size = subset_size
        assert 0 < self.subset_size <= self.population_size
        self.epsilon = epsilon
        self.strategy = strategy
        assert self.strategy in ['worst', 'oldest']

    def get_suggestion(self):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """

        # Removed the parameter.
        """
        if history_container is None:
            history_container = self.history_container
        """

        if len(self.population) < self.population_size:
            # Initialize population
            next_config = self.sample_random_config(excluded_configs=self.all_configs)
        else:
            # Select a parent by subset tournament and epsilon greedy
            if self.rng.random() < self.epsilon:
                parent_config = random.sample(self.population, 1)[0]['config']
            else:
                subset = random.sample(self.population, self.subset_size)

                subset.sort(key=lambda x: x['perf'])  # minimize
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
        perf = observation.objs[0]
        constraint = constraint_check(observation.constraints)
        trial_state = observation.trial_state

        assert config in self.running_configs
        self.running_configs.remove(config)

        # update population
        if trial_state == SUCCESS and perf < MAXINT:
            self.population.append(Individual(config=config, age=self.age, perf=perf, constraints_satisfied=constraint))
            self.age += 1

        if not constraint:
            return self.history_container.update_observation(observation)
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

        return self.history_container.update_observation(observation)
