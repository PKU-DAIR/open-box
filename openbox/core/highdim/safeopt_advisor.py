import random
import typing
from copy import deepcopy
from typing import Optional, Callable, List, Type, Union, Any, Tuple, Dict

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, CategoricalHyperparameter, \
    OrdinalHyperparameter

from openbox import Advisor
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.acquisition_function import AbstractAcquisitionFunction
from openbox.core.base import build_acq_func, build_surrogate, Observation, build_optimizer
from openbox.core.ea.base_ea_advisor import Individual
from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor
from openbox.core.highdim.linebo_advisor import LineBOAdvisor
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.config_space import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer, MOHistoryContainer
from openbox.utils.multi_objective import NondominatedPartitioning, get_chebyshev_scalarization
from openbox.utils.util_funcs import check_random_state


class SafeOptWrapper:

    def __init__(self,
                 h=0.01):
        self.h = h

    def __call__(self, observation: Observation) -> Observation:
        x = observation.objs[0]
        observation1 = Observation(observation.config, observation.objs,
                                   [x - self.h] if observation.constraints is None else observation.constraints + [
                                       x - self.h]
                                   , observation.trial_state, observation.elapsed_time)
        return observation1


class SafeOptAdvisor:

    def __init__(self, config_space: ConfigurationSpace,
                 num_objs=1,
                 num_constraints=0,
                 task_id='default_task_id',
                 random_state=None,

                 sub_advisor: Union[Type, Tuple[Type, Tuple, Dict]] = LineBOAdvisor,

                 h: float = 0.01

                 ):
        self.num_objs = num_objs
        # May support multi-obj in the future.
        assert self.num_objs == 1

        self.num_constraints = num_constraints

        self.config_space = config_space
        self.dim = len(config_space.keys())
        self.rng = check_random_state(random_state)
        self.task_id = task_id

        if isinstance(sub_advisor, Tuple):
            self.sub_advisor = sub_advisor[0](config_space, *sub_advisor[1], num_objs=num_objs,
                                              num_constraints=num_constraints + 1,
                                              random_state=random_state, **sub_advisor[2])
        else:
            self.sub_advisor = sub_advisor(config_space, num_objs=num_objs, num_constraints=num_constraints + 1,
                                           random_state=random_state)

        self.wrapper = SafeOptWrapper(h=h)

        self.history_container = HistoryContainer(task_id, num_constraints, self.config_space) if num_objs == 1 else \
            MOHistoryContainer(task_id, num_objs, num_constraints, self.config_space)

    def get_suggestion(self):
        return self.sub_advisor.get_suggestion()

    def update_observation(self, observation: Observation):
        self.sub_advisor.update_observation(self.wrapper(observation))

        return self.history_container.update_observation(observation)

    def get_suggestions(self, batch_size=None):
        return self.sub_advisor.get_suggestions(batch_size)

    def update_observations(self, observations: List[Observation]):
        return [self.update_observation(o) for o in observations]

    def get_history(self):
        return self.history_container
