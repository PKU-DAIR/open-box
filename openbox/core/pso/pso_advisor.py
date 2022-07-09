import abc

import random

from typing import *

import numpy as np
from openbox.core.pso.base_pso_advisor import Individual
from openbox.core.pso.base_pso_advisor import BasePSOAdvisor
from openbox.utils.util_funcs import check_random_state
from openbox.utils.logging_utils import get_logger
from openbox.utils.history_container import HistoryContainer, MOHistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.core.base import Observation

from ConfigSpace import ConfigurationSpace, Configuration


class PSOAdvisor(BasePSOAdvisor):
    def __init__(self, config_space: ConfigurationSpace,
                 num_objs = 1,
                 num_constraints = 0,
                 population_size = 10,
                 batch_size = 1,
                 output_dir = 'logs',
                 task_id = 'default_task_id',
                 random_state = None,

                 det = 1,
                 wi = 0.729,
                 c1 = 1.3,
                 c2 = 1.3,
                 ):

        super().__init__(config_space = config_space, num_objs = num_objs, num_constraints = num_constraints,
                         population_size = population_size, batch_size = batch_size, output_dir = output_dir,
                         task_id = task_id, random_state = random_state,
                         )

        # PSO params
        self.gbest = None
        self.d_len = len(self.config_space.keys())
        self.det = det
        self.wi = wi
        self.c1 = c1
        self.c2 = c2
        self.curr = 1
        self.vel_max = 0.5
        self.pbest: List[Union[Dict, Individual]] = list()

    def get_suggestions(self):
        configs = []
        next_vel = []
        next_config = []
        if len(self.population) == 0:
            for t in range(self.population_size):
                next_config = self.sample_random_config(self.config_space, excluded_configs = self.all_configs)
                next_vel = np.random.uniform(-self.vel_max, self.vel_max, size = (1, self.d_len))[0]
                configs.append(next_config)
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
                # print(next_config, next_vel)
                self.population.append(Individual(pos = next_config.get_array(),
                                                  vel = next_vel, perf = MAXINT))
                self.pbest.append(Individual(pos = next_config.get_array(),
                                             vel = next_vel, perf = MAXINT))
            self.gbest = Individual(pos = next_config.get_array(),
                                    vel = next_vel, perf = MAXINT)
        else:
            for t in self.population:
                next_config = Configuration(self.config_space, vector = list(t['pos']))
                configs.append(next_config)
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
        # print(configs)
        return configs

    def update_observations(self, observations: [Observation]):
        for t in range(self.population_size):
            observation = observations[t]
            config = observation.config
            trial_state = observation.trial_state
            assert config in self.running_configs
            self.running_configs.remove(config)

            perf = observation.objs[0]
            self.population[t]['perf'] = perf
            self.history_container.update_observation(observation)

            for i in range(self.population_size):
                cur = self.population[i]
                curbest = self.pbest[i]
                if cur['perf'] < curbest['perf']:
                    curbest['pos'] = cur['pos']
                    curbest['perf'] = cur['perf']
                if cur['perf'] < self.gbest['perf']:
                    self.gbest['pos'] = cur['pos']
                    self.gbest['perf'] = cur['perf']

            if self.gbest['perf'] != MAXINT:
                for i in range(self.population_size):
                    cur = self.population[i]
                    curbest = self.pbest[i]
                    cur['vel'] = self.update_vel(cur['pos'], cur['vel'], curbest, self.gbest)
                    cur['pos'] = self.update_pos(cur['pos'], cur['vel'])
                self.curr *= self.det

            else:
                for i in range(self.population_size):
                    cur = self.population[i]
                    cur['pos'] = self.update_pos(cur['pos'], cur['vel'])

    def update_vel(self, pos: np.ndarray, vel: np.ndarray, pbest: Individual, gbest: Individual):
        pbest = pbest['pos']
        gbest = gbest['pos']
        r1 = np.random.random()
        r2 = np.random.random()
        vel = self.wi * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
        vel = self.curr * vel
        vel[vel > self.vel_max] = self.vel_max
        vel[vel < -self.vel_max] = -self.vel_max
        # print("-----", vel)
        return vel

    def update_pos(self, pos: np.ndarray, vel: np.ndarray):
        pos = pos + vel
        return pos

