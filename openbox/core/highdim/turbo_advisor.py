import random
import typing
from typing import Optional, Callable, List, Type, Union

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, CategoricalHyperparameter, \
    OrdinalHyperparameter
from skopt.sampler import Sobol

from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.acquisition_function import AbstractAcquisitionFunction
from openbox.core.base import build_acq_func, build_surrogate, Observation, build_optimizer
from openbox.core.ea.base_ea_advisor import Individual
from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.config_space import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer
from openbox.utils.multi_objective import NondominatedPartitioning, get_chebyshev_scalarization
from openbox.utils.util_funcs import check_random_state


class TuRBOAdvisor:

    def __init__(self, config_space: ConfigurationSpace,
                 num_objs=1,
                 num_constraints=0,
                 task_id='default_task_id',
                 batch_size=10,
                 random_state=None,

                 surrogate: str = 'gp_rbf',

                 rand_count=5,
                 success_limit=3,
                 fail_limit=3,

                 size_initial=0.125,
                 size_max=0.5,
                 size_min=0.5 ** 7,

                 candidate_count=5000,
                 candidate_select_count=5
                 ):
        self.num_objs = num_objs
        # Does not support multi-obj!
        assert self.num_objs == 1
        # Does not support constraints!
        self.num_constraints = num_constraints
        assert self.num_constraints == 0

        self.config_space = config_space
        self.dim = len(config_space.keys())
        self.batch_size = batch_size
        self.rng = check_random_state(random_state)
        self.task_id = task_id

        self.objective_surrogate: AbstractModel = build_surrogate(surrogate, config_space, self.rng or random, None)

        self.history_container = HistoryContainer(task_id, num_constraints, self.config_space)

        self.rand_count = rand_count
        self.success_limit = success_limit
        self.fail_limit = fail_limit

        self.cnt = 0
        self.success_count = 0
        self.fail_count = 0

        self.region_size_min = size_min
        self.region_size_max = size_max
        self.region_size = size_initial

        # TODO gp-rbf always have length_scale = 1. For other gp models, how can we find its length_scale?
        self.length_scale = 1
        self.candidate_count = candidate_count
        self.candidate_select_count = candidate_select_count

        self.incumbent = 1e100
        self.incumbent_config: Optional[Configuration] = None

        self.sampled_configs = []
        self.inner_loop = False

        self.latin_hypercube_sample()

    def latin_hypercube_sample(self, count=None):
        """
        Return:

        np.ndarray (count, self.dim)
        """
        if count is None:
            count = self.rand_count

        layered_dim = self.rng.randint(0, self.dim - 1)

        res = self.rng.random((count, self.dim))
        for i in range(count):
            res[i, layered_dim] = (res[i, layered_dim] + i) * (1 / count)

        self.sampled_configs = [Configuration(self.config_space, vector=res[i]) for i in range(count)]

    def adjust_length(self, perf: float):
        if perf < self.incumbent * 0.9999:
            self.success_count += 1
            self.fail_count = 0
        else:
            self.success_count = 0
            self.fail_count += 1

        if self.success_count == self.success_limit:
            self.region_size = min(self.region_size * 2, self.region_size_max)
            self.success_count = 0

        if self.fail_count == self.fail_limit:
            self.region_size = self.region_size / 2
            self.fail_count = 0

    def create_candidates(self):

        X = convert_configurations_to_array(self.history_container.configurations)
        Y = self.history_container.get_transformed_perfs(transform=None)
        self.objective_surrogate.train(X, Y[:, 0] if Y.ndim == 2 else Y)

        w = self.length_scale

        x_center = self.incumbent_config.get_array()

        lb = np.clip(x_center - w * self.region_size / 2, 0.0, 1.0)
        ub = np.clip(x_center + w * self.region_size / 2, 0.0, 1.0)

        # generate sobol sequence
        skip_count = int(2 ** int(np.log2(self.candidate_count) + 1))
        sobol = Sobol()
        sobol.init(dim_num=self.dim)

        XC = np.array(sobol.generate([(lb[i], ub[i]) for i in range(lb.shape[0])], skip_count)[:self.candidate_count])

        YC = self.objective_surrogate.predict(XC)[0][:, 0]

        return XC, YC

    def select_candidates(self, XC, YC):
        # Strategy: select the bests
        Yranks = np.argsort(YC)
        return XC[Yranks[:self.candidate_select_count]]

    def get_suggestion(self):
        if len(self.sampled_configs) != 0:
            return self.sampled_configs[0]

    def update_observation(self, observation: Observation):
        self.history_container.update_observation(observation)

        config = observation.config
        perf = observation.objs[0]

        if perf < self.incumbent:
            self.incumbent = perf
            self.incumbent_config = config

        if config in self.sampled_configs:
            self.sampled_configs.remove(config)

            if len(self.sampled_configs) == 0:

                do_remaining = False

                if self.inner_loop:  # Just finished evaluation of trust-region sampled configs
                    self.adjust_length(perf)

                    if self.region_size <= self.region_size_min:
                        # exit inner loop, randomly sample points
                        self.latin_hypercube_sample()
                        self.inner_loop = False
                    else:
                        do_remaining = True
                else:  # Just finished evaluation of latin-hypercube sampled configs
                    self.inner_loop = True
                    do_remaining = True

                if do_remaining:  # Inner loop begin -> evaluation
                    XC, YC = self.create_candidates()
                    XS = self.select_candidates(XC, YC)

                    self.sampled_configs = [Configuration(self.config_space, vector=XS[i]) for i in range(XS.shape[0])]

    def get_suggestions(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.sampled_configs) != 0:
            return self.sampled_configs[:min(len(self.sampled_configs), batch_size)]
        else:
            return [self.get_suggestion()]

    def update_observations(self, observations: List[Observation]):
        return [self.update_observation(o) for o in observations]

    def get_history(self):
        return self.history_container
