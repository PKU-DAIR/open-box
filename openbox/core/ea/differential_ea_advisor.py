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


from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import *

from typing import *


class DifferentialEAAdvisor(EAAdvisor):

    def __init__(self,
                 f: Union[Tuple[float, float], float] = 0.5,
                 cr: Union[Tuple[float, float], float] = 0.8,
                 **kwargs):
        """
        f is the hyperparameter for DEA that X = A + (B - C) * f
        cr is the cross rate
        f and cr may be a tuple of two floats, such as (0.1,0.9)
        If so, these two values are adjusted automatically within this range.
        """
        EAAdvisor.__init__(self, **kwargs)

        self.f = f
        self.cr = cr

        self.iter = None
        self.cur = 0

        self.running_origin_map = dict()

        self.next_population = [None for i in range(self.population_size)]


    def get_suggestion(self):


        if len(self.population) < self.population_size:
            next_config = self.sample_random_config(excluded_configs=self.all_configs)
            nid = -1

        else:

            if self.cur == 0:
                for i in range(self.population_size):
                   if self.next_population[i] is not None:
                       if self.next_population[i]['perf'] < self.population[i]['perf']:
                           self.population[i] = self.next_population[i]
                # print(self.population)

            # Run one iteration of DEA if the population is filled.

            # xi is the current value.
            xi = self.population[self.cur]['config']
            xi_score = self.population[self.cur]['perf']

            # Randomly sample 3 other values: x1, x2, x3
            lst = list(range(self.population_size))
            lst.remove(self.cur)
            random.shuffle(lst)
            lst = lst[:3]

            if isinstance(self.f, tuple):
                lst.sort(key=lambda a: self.population[a]['perf'])

            i1, i2, i3 = lst[0], lst[1], lst[2]
            x1, x2, x3 = self.population[i1]['config'], self.population[i2]['config'], self.population[i3]['config']

            # Mutation: xt = x1 + (x2 - x3) * f
            if isinstance(self.f, tuple):
                # Dynamic f
                f1, f2, f3 = self.population[i1]['perf'], self.population[i2]['perf'], self.population[i3]['perf']
                if f1 == f3:
                    f = self.f[0]
                else:
                    f = self.f[0] + (self.f[1] - self.f[0]) * (f1 - f2) / (f1 - f3)
            else:
                # Fixed f
                f = self.f

            xt = self.mutate(x1, x2, x3, f)

            # Cross over between xi and xt, get xn
            if isinstance(self.cr, tuple):
                # Dynamic cr
                scores = [a['perf'] for a in self.population]
                scores_avg = sum(scores) / len(scores)

                if xi_score < scores_avg:
                    scores_mx = max(scores)
                    scores_mn = min(scores)
                    cr = self.cr[0] + (self.cr[1] - self.cr[0]) * (scores_mx - xi_score) / max(
                            scores_mx - scores_mn, 1e-10)
                else:
                    cr = self.cr[0]
            else:
                # Fixed cr
                cr = self.cr

            xn = self.cross_over(xi, xt, cr)

            # xn should be evaluated.
            # if xn is better than xi, we replace xi with xn.

            # xn = get_one_exchange_neighbourhood(xi)

            next_config = xn
            nid = self.cur
            self.cur = (self.cur + 1) % self.population_size

        self.all_configs.add(next_config)
        # nid keeps track of which xi that the xn should compare with.
        # nid == -1 indicates that this xn is randomly sampled.
        self.running_configs.append((next_config, nid))

        # print(self.x, next_config.get_array())

        return next_config

    def update_observation(self, observation: Observation):

        config = observation.config
        perf = observation.objs[0]
        trial_state = observation.trial_state

        nid = [p[1] for p in self.running_configs if p[0] == config]

        assert len(nid) == 1
        nid = nid[0]

        self.running_configs.remove((config,nid))

        if trial_state == SUCCESS and perf < MAXINT:
            if nid == -1 and len(self.population) < self.population_size:
                # The population has not yet been filled.
                self.population.append(dict(config=config, age=self.age, perf=perf))
            elif nid == -1:
                # The population has been filled, but more configs are generated while it was not filled.
                # In this case, simply replace the worst one with it.
                self.population.append(dict(config=config, age=self.age, perf=perf))
                self.population.sort(key=lambda x: x['perf'])
                self.population.pop(-1)
            else:
                # Compare xn with xi. If xn is better, replace xi with it.
                self.next_population[nid] = dict(config=config, age=self.age, perf=perf)

        return self.history_container.update_observation(observation)

    def mutate(self, config_a: Configuration, config_b: Configuration, config_c: Configuration, f: float):
        """
        Compute A + (B - C) * f. Basically element-wise.
        For ranged int/float values, the result will be clamped into [lower, upper].
        For categorical/ordinal values, the values are converted to ints and the result is (mod SIZE).
        e. g. in ["A", "B", "C", "D"], "D" + "B" - "A" => 3 + 1 - 0 => 4 => 0 (mod 4) => "A"
        """
        new_array = config_a.get_array() + (config_b.get_array() - config_c.get_array()) * f

        for i, key in enumerate(self.config_space.keys()):
            hp_type = self.config_space.get_hyperparameter(key)
            if isinstance(hp_type, CategoricalHyperparameter) or isinstance(hp_type, OrdinalHyperparameter):
                v = (round(new_array[i]) % hp_type.get_size() + hp_type.get_size()) % hp_type.get_size()
                new_array[i] = v
            elif isinstance(hp_type, NumericalHyperparameter):
                # new_array[i] = max(0, min(new_array[i], 1))
                if new_array[i] < 0:
                    # new_array[i] = -new_array[i] / 2
                    new_array[i] = random.random()
                if new_array[i] > 1:
                    # new_array[i] = 1 - (new_array[i] - 1) / 2
                    new_array[i] = random.random()
            else:
                pass

        config = Configuration(self.config_space, vector=new_array)
        return config

    def cross_over(self, config_a: Configuration, config_b: Configuration, cr: float):
        """
        The cross-over operation.
        For each element of config_a, it has cr possibility to be replaced with that of config_b.
        """
        a1, a2 = config_a.get_array(), config_b.get_array()
        any_changed = False

        for i in range(len(self.config_space.keys())):
            if self.rng.random() < cr:
                a1[i] = a2[i] # a1, a2 are vector copies, modification is ok.
                any_changed = True

        # Make sure cross-over changes at least one dimension.
        if not any_changed:
            i = self.rng.randint(0, len(self.config_space.keys()) - 1)
            a1[i] = a2[i]

        return Configuration(self.config_space, vector=a1)
