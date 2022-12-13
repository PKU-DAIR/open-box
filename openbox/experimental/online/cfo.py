from typing import List, Optional

from ConfigSpace import ConfigurationSpace, Configuration

from openbox.utils.history import Observation
from openbox.experimental.online.base_online_advisor import OnlineAdvisor


class CFO(OnlineAdvisor):

    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration,
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,

                 inc_threshould = 20,
                 delta_init: float = 0.05,
                 delta_lower: float = 0.002,
                 noise_scale: float = 0.1
                 ):
        super().__init__(config_space=config_space, x0=x0, batch_size=batch_size, output_dir=output_dir,
                         task_id=task_id, random_state=random_state)
        self.delta = delta_init
        self.delta_init = delta_init
        self.delta_lower = delta_lower
        self.dim = len(config_space.keys())

        self.noise_scale = noise_scale

        self.x = x0
        self.config = None

        self.conf: List[Configuration] = []
        self.res: List[Optional[float]] = [None] * 3

        self.refresh = True
        self.k = self.kd = self.n = self.r = 0
        self.lr_best = 1e100

        self.inc = 1e100
        self.incn = 0
        self.inc_threshould = inc_threshould

    def get_suggestion(self):

        if all(x is not None for x in self.res):
            r0 = self.res[0]
            if self.res[1] < self.res[0]:
                self.x = self.conf[1]
                self.res = [self.res[1], None, None]
            elif self.res[2] < self.res[0]:
                self.x = self.conf[2]
                self.res = [self.res[2], None, None]
            else:
                self.n += 1
                self.res = [self.res[0], None, None]

            self.k += 1

            if r0 < self.lr_best:
                self.lr_best = r0
                self.kd = self.k

            if self.n == 2 ** (self.dim - 1):
                self.n = 0
                self.delta = self.delta * (1 / (self.k / self.kd) ** 0.5)
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
            self.refresh = False

        # print(self.conf)

        for i in range(3):
            if self.res[i] is None:
                self.config = self.conf[i]
                return self.conf[i]

    def update_observation(self, observation: Observation):
        self.history.update_observation(observation)

        if observation.objectives[0] < self.inc:
            self.inc = observation.objectives[0]
            self.incn = 0
        else:
            self.incn += 1

        for i in range(3):
            if observation.config == self.conf[i] and self.res[i] is None:
                self.res[i] = observation.objectives[0]
                break

    def is_converged(self):
        return self.incn > self.inc_threshould
