import abc
from typing import Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox.core.base import Observation
from openbox.utils.history_container import HistoryContainer
from openbox.utils.util_funcs import check_random_state


def almost_equal(config1: Configuration, config2: Configuration, delta: float = 1e-4):
    if not (config1 and config2):
        return False
    return np.linalg.norm(np.abs(config1.get_array() - config2.get_array())) < delta


class OnlineAdvisor(abc.ABC):
    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration,
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None):
        self.config_space = config_space
        self.x0 = x0
        self.config: Configuration
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.rng = check_random_state(random_state)

        self.history_container = HistoryContainer(task_id, 0, self.config_space)

    def get_suggestion(self):
        raise NotImplementedError

    def update_observation(self, observation: Observation):
        raise NotImplementedError

    def is_converged(self):
        raise NotImplementedError

    def get_history(self):
        return self.history_container

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
                arr[i] = self.rng.randint(0, hp_type.get_size() - 1)
                arr1[i] = self.rng.randint(0, hp_type.get_size() - 1)
            elif isinstance(hp_type, NumericalHyperparameter):
                arr[i] = min(arr[i] + d[i], 1.0)
                arr1[i] = max(arr1[i] - d[i], 0.0)
            else:
                pass

        # print(arr)
        # print(arr1)

        return Configuration(self.config_space, vector=arr), Configuration(self.config_space, vector=arr1)
