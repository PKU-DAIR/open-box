from typing import List

from ConfigSpace import ConfigurationSpace, Configuration

from openbox.utils.history import Observation
from openbox.experimental.online.base_online_advisor import OnlineAdvisor


class FLOW2(OnlineAdvisor):

    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration=None,
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,

                 inc_threshould = 20,
                 delta: float = 0.01
                 ):
        super().__init__(config_space=config_space, x0=x0, batch_size=batch_size, output_dir=output_dir,
                         task_id=task_id, random_state=random_state)
        self.delta = delta
        self.dim = len(config_space.keys())

        self.x = x0 or config_space.sample_configuration()
        self.config = None
        self.conf: List[Configuration] = []
        self.res = [None] * 3
        self.refresh = True

        self.inc = 1e100
        self.incn = 0
        self.inc_threshould = inc_threshould

    def get_suggestion(self):
        if self.res[1] is not None and self.res[0] is not None:
            if self.res[1] < self.res[0]:
                self.x = self.conf[1]
                self.res = [self.res[1], None, None]
                self.refresh = True

        if all(x is not None for x in self.res):
            if self.res[2] < self.res[0]:
                self.x = self.conf[2]
                self.res = [self.res[2], None, None]
            else:
                self.res = [self.res[0], None, None]
            self.refresh = True

        if self.refresh:
            x1, x2 = self.next(self.x, self.delta)
            self.conf = [self.x, x1, x2]
            self.refresh = False

        for i in range(3):
            if self.res[i] is None:
                self.config = self.conf[i]
                return self.conf[i]

    def update_observation(self, observation: Observation):
        self.history.update_observation(observation)

        for i in range(3):
            if observation.config == self.conf[i] and self.res[i] is None:
                self.res[i] = observation.objectives[0]
                if observation.objectives[0] < self.inc:
                    self.inc = observation.objectives[0]
                    self.incn = 0
                else:
                    self.incn += 1
                break

    def is_converged(self):
        return self.incn > self.inc_threshould
