# License: MIT
from typing import List

from ConfigSpace import Configuration

from openbox.core.ea.base_ea_advisor import Individual
from openbox.core.ea.base_ea_advisor import pareto_layers
from openbox.core.ea.base_ea_advisor import EAAdvisor
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.config_space import get_one_exchange_neighbourhood
from openbox.utils.history import Observation
from openbox.utils.util_funcs import deprecate_kwarg


class NSGA2EAdvisor(EAAdvisor):
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
                 strategy='worst',
                 ):

        super().__init__(config_space=config_space, num_objectives=num_objectives, num_constraints=num_constraints,
                         population_size=population_size, optimization_strategy=optimization_strategy,
                         batch_size=batch_size, output_dir=output_dir, task_id=task_id, random_state=random_state,
                         )

        self.subset_size = subset_size
        assert 0 < self.subset_size <= self.population_size
        self.epsilon = epsilon
        self.strategy = strategy
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
            for t in range(miu):
                next_config = self.sample_random_config(excluded_configs=self.all_configs)
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
                next_configs.append(next_config)
        else:
            # mutation here
            for cur in self.population:
                next_config = self.mutation(cur['config'])
                self.all_configs.add(next_config)
                self.running_configs.append(next_config)
                next_configs.append(next_config)

            # cross_over here
            for nu in range(self.population_size):
                nv = self.population_size - nu - 1
                if nu >= nv:
                    break
                parent0, parent1 = self.population[nu]['config'], self.population[nv]['config']
                next_config = self.cross_over(parent0, parent1)
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
            perf = observation.objectives
            trial_state = observation.trial_state
            assert config in self.running_configs
            self.running_configs.remove(config)

            # update population
            if trial_state == SUCCESS:
                self.population.append(Individual(config=config, perf=perf))

            ret_observations.append(self.history.update_observation(observation))

        # Eliminate samples
        if len(self.population) > self.population_size:
            layers = pareto_layers(self.population)
            laynum = len(layers)
            tot = 0
            self.population = []
            for t in range(laynum):
                if tot + len(layers[t]) > self.population_size:
                    miu = self.population_size - tot
                    self.population += self.crowding_select(layers[t], miu)
                    break
                else:
                    self.population += layers[t]
                    tot += len(layers[t])

        return ret_observations

    def cross_over(self, config_a: Configuration, config_b: Configuration):
        a1, a2 = config_a.get_array(), config_b.get_array()
        s_len = len(self.config_space.keys())
        for i in range(s_len):
            if s_len < 3:
                a1[i] = (a1[i] + a2[i]) * 0.5
            else:
                if self.rng.random() < 0.5:
                    a1[i] = a2[i]

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

    def crowding_select(self, xs: List[Individual], num) -> List[Individual]:
        INF = 1e7
        llen = len(xs)
        dim = len(xs[0]['perf'])
        xs = [[x, 0] for x in xs]
        for k in range(dim):
            xs = sorted(xs, key=lambda xv: xv[0]['perf'][k], reverse=True)
            xs[0][1] += INF
            xs[-1][1] += INF
            for t in range(1, llen - 1):
                xs[t][1] += xs[t - 1][0]['perf'][k] - xs[t + 1][0]['perf'][k]
        xs = sorted(xs, key=lambda xv: xv[1], reverse=True)
        xs = xs[:num]
        return [x for (x, v) in xs]
