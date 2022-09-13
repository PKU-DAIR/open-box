from typing import Tuple, List, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox import Observation
from openbox.core.online.utils.base_searcher import Searcher


class FLOW2(Searcher):

    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration,
                 delta_init: float = 1.0,
                 delta_lower: float = 0.01,
                 noise_scale: float = 0.1
                 ):
        super().__init__(config_space=config_space, x0=x0)
        self.delta = delta_init
        self.delta_init = delta_init
        self.delta_lower = delta_lower
        self.dim = len(config_space.keys())

        self.noise_scale = noise_scale

        self.x = x0

        self.conf: List[Configuration] = []
        self.res: List[Optional[float]] = [None] * 3

        self.refresh = True
        self.k = self.kd = self.n = self.r = 0
        self.lr_best = 1e100

    def get_suggestion(self):

        if all(self.res):
            if self.res[1] < self.res[0]:
                self.x = self.conf[1]
            elif self.res[2] < self.res[0]:
                self.x = self.conf[2]
            else:
                self.n += 1

            if self.res[0] < self.lr_best:
                self.lr_best = self.res[0]
                self.kd = self.k

            if self.n == 2 ** (self.dim - 1):
                self.n = 0
                self.delta = self.delta / (1 / (self.k / self.kd) ** 0.5)
                if self.delta <= self.delta_lower:
                    self.k = 0
                    self.lr_best = 1e100
                    self.x = self.next(self.x0, self.noise_scale, True)[0]
                    self.r += 1
                    self.delta = self.r + self.delta_init

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
