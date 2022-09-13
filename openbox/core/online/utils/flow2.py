from typing import Tuple, List

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox import Observation
from openbox.core.online.utils.base_searcher import Searcher


class FLOW2(Searcher):

    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration,
                 delta: float = 0.01
                 ):
        super().__init__(config_space=config_space, x0=x0)
        self.delta = delta
        self.dim = len(config_space.keys())

        self.x = x0
        self.conf: List[Configuration] = []
        self.res = [None] * 3
        self.refresh = True


    def get_suggestion(self):
        if self.res[1] and self.res[0]:
            if self.res[1] < self.res[0]:
                self.x = self.conf[1]
                self.refresh = True

        if all(self.res):
            if self.res[2] < self.res[0]:
                self.x = self.conf[2]
            self.refresh = True

        if self.refresh:
            x1, x2 = self.next(self.x, self.delta)
            self.conf = [self.x, x1, x2]
            self.res = [None] * 3
            self.refresh = False

        for i in range(3):
            if not self.res[i]:
                return self.conf[i]

    def update_observation(self, observation: Observation):
        self.history_container.update_observation(observation)

        for i in range(3):
            if observation.config == self.conf[i]:
                self.res[i] = observation.objs[0]
                break
