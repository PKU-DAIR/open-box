# License: MIT
from typing import List
import numpy as np

from ConfigSpace import Configuration

from openbox.core.ea.base_ea_advisor import EAAdvisor
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.config_space import get_one_exchange_neighbourhood
from openbox.utils.util_funcs import deprecate_kwarg


class AdaptiveEAAdvisor(EAAdvisor):

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self,

                 config_space,
                 num_objectives=1,
                 num_constraints=0,
                 population_size=30,
                 optimization_strategy='ea',
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,

                 subset_size=20,
                 epsilon=0.2,
                 pc=0.3,
                 pm=0.3,
                 strategy='worst',
                 ):

        super().__init__(config_space, num_objectives=num_objectives, num_constraints=num_constraints,
                         population_size=population_size, optimization_strategy=optimization_strategy,
                         batch_size=batch_size, output_dir=output_dir, task_id=task_id, random_state=random_state,
                         )

        self.subset_size = subset_size
        assert 0 < self.subset_size <= self.population_size
        self.epsilon = epsilon
        self.pm = pm
        self.pc = pc
        self.strategy = strategy
        self.k1, self.k2, self.k3, self.k4 = 0.25, 0.3, 0.25, 0.3
        assert self.strategy in ['worst', 'oldest']

        self.last_suggestions = []
        self.last_observations = []

    def get_suggestion(self):
        if not self.last_suggestions:
            self.update_observations(self.last_observations)
            self.last_observations = []
            self.last_suggestions = self.get_suggestions()
        return self.last_suggestions.pop()

    def get_suggestions(self, batch_size=None):
        next_configs = []
        if len(self.population) < self.population_size:
            miu = self.population_size - len(self.population)
            for t in range(0, miu):
                next_config = self.sample_random_config(excluded_configs=self.all_configs)
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
                next_configs.append(next_config)
        else:
            all_perf = [cur['perf'] for cur in self.population]
            fmax = max(all_perf)
            favg = np.mean(all_perf)
            # cross_over here
            for nu in range(0, self.population_size):
                nv = self.population_size - nu - 1
                if nu > nv:
                    break

                # Adaptive pc here
                fm = max(self.population[nu]['perf'], self.population[nv]['perf'])
                if fm > favg:
                    self.pc = self.k1 * (fmax - fm) / (fmax - favg)
                else:
                    self.pc = self.k2

                if self.rng.random() < self.pc:
                    parent0, parent1 = self.population[nu]['config'], self.population[nv]['config']
                    next_config = self.cross_over(parent0, parent1)
                    self.all_configs.add(next_config)
                    self.running_configs.append(next_config)
                    next_configs.append(next_config)

            # mutation here
            for cur in self.population:

                # Adaptive pm here
                fm = cur['perf']
                if fm > favg:
                    self.pm = self.k3 * (fmax - fm) / (fmax - favg)
                else:
                    self.pm = self.k4

                if self.rng.random() < self.pm:
                    next_config = self.mutation(cur['config'])
                    self.all_configs.add(next_config)
                    self.running_configs.append(next_config)
                    next_configs.append(next_config)
        return next_configs

    def update_observation(self, observation: Observation):
        self.last_observations.append(observation)

    def update_observations(self, observations: List[Observation]):
        self.age += 1
        ret_observations = []
        for observation in observations:
            config = observation.config
            perf = observation.objectives[0]
            trial_state = observation.trial_state
            assert config in self.running_configs
            self.running_configs.remove(config)

            # update population
            if trial_state == SUCCESS and perf < np.inf:
                self.population.append(dict(config=config, age=self.age, perf=perf))

            ret_observations.append(self.history.update_observation(observation))

        # Eliminate samples
        if len(self.population) > self.population_size:
            if self.strategy == 'oldest':
                self.population.sort(key=lambda x: x['age'])
                self.population = self.population[-self.population_size:]
            elif self.strategy == 'worst':
                self.population.sort(key=lambda x: x['perf'])
                self.population = self.population[:self.population_size]
            else:
                raise ValueError('Unknown strategy: %s' % self.strategy)

        return ret_observations

    def cross_over(self, config_a: Configuration, config_b: Configuration):
        a1, a2 = config_a.get_array(), config_b.get_array()

        for i in range(len(self.config_space.keys())):
            a1[i] = (a1[i] + a2[i]) * 0.5
            # if self.rng.random() < 0.5:
            #     a1[i] = a2[i]
            # a1[i] = a1[i] * (1.0 - cr) + a2[i] * cr

        return Configuration(self.config_space, vector=a1)

    def mutation(self, config: Configuration):
        ret_config = None
        neighbors_gen = get_one_exchange_neighbourhood(config, seed=self.rng.randint(MAXINT))
        for neighbor in neighbors_gen:
            if neighbor not in self.all_configs:
                ret_config = neighbor
                break
        if ret_config is None:
            ret_config = self.sample_random_config(excluded_configs=self.all_configs)
        return ret_config
