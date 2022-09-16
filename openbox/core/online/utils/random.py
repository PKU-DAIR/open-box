from typing import List

from ConfigSpace import ConfigurationSpace, Configuration

from openbox import Observation
from openbox.core.online.utils.base_searcher import Searcher


class RandomSearch(Searcher):

    def is_converged(self):
        return False

    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration,
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None,
                 ):
        super().__init__(config_space=config_space, x0=x0, batch_size=batch_size, output_dir=output_dir,
                         task_id=task_id, random_state=random_state)
        self.dim = len(config_space.keys())

        self.x = x0
        self.config = None

    def get_suggestion(self):
        self.config = self.config_space.sample_configuration()
        return self.config

    def update_observation(self, observation: Observation):
        self.history_container.update_observation(observation)
