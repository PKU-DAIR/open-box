from typing import List, Optional

from ConfigSpace import ConfigurationSpace, Configuration

from openbox import Observation
from openbox.core.online.utils.base_online_advisor import OnlineAdvisor


class BOOnlineAdvisor(OnlineAdvisor):

    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration,
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
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

        pass


    def update_observation(self, observation: Observation):
        self.history_container.update_observation(observation)

        if observation.objs[0] < self.inc:
            self.inc = observation.objs[0]
            self.incn = 0
        else:
            self.incn += 1

        for i in range(3):
            if observation.config == self.conf[i] and self.res[i] is None:
                self.res[i] = observation.objs[0]
                break

    def is_converged(self):
        return self.incn > self.inc_threshould
