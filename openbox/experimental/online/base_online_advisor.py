import abc
from typing import Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox.utils.history import Observation, History
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg


def almost_equal(config1: Configuration, config2: Configuration, delta: float = 1e-4):
    if not (config1 and config2):
        return False
    return np.linalg.norm(np.abs(config1.get_array() - config2.get_array())) < delta


class OnlineAdvisor(abc.ABC):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration,
                 num_objectives=1,
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None):
        self.config_space = config_space
        self.x0 = x0
        self.num_objectives = num_objectives
        self.config: Configuration
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.rng = check_random_state(random_state)

        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=0, config_space=config_space,
            ref_point=None, meta_info=None,  # todo: add ref_point, meta info
        )

    def get_suggestion(self):
        raise NotImplementedError

    def update_observation(self, observation: Observation):
        raise NotImplementedError

    def is_converged(self):
        raise NotImplementedError

    def get_history(self):
        return self.history

    def next(self, config_a: Configuration, delta: float, gaussian=False) -> Tuple[Configuration, Configuration]:
        """
        Given x, delta, sample u randomly from unit sphere, or N(0, 1) if gaussian is True.
        return (x + delta * u, x - delta * u).
        Chooses another random value for categorical hyper-parameters.
        """

        arr = config_a.get_array().copy()
        arr1 = arr.copy()

        # print("--", arr)

        d = np.random.randn(*arr.shape)
        if not gaussian:
            d = d / np.linalg.norm(d)
        d = d * delta

        # print(d)

        for i, key in enumerate(self.config_space.keys()):
            hp_type = self.config_space.get_hyperparameter(key)
            if isinstance(hp_type, CategoricalHyperparameter) or isinstance(hp_type, OrdinalHyperparameter):
                arr[i] = self.rng.randint(0, hp_type.get_size())
                arr1[i] = self.rng.randint(0, hp_type.get_size())
            elif isinstance(hp_type, NumericalHyperparameter):
                arr[i] = min(arr[i] + d[i], 1.0)
                arr1[i] = max(arr1[i] - d[i], 0.0)
            else:
                pass

        # print(arr)
        # print(arr1)

        return Configuration(self.config_space, vector=arr), Configuration(self.config_space, vector=arr1)
