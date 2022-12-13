from ConfigSpace import ConfigurationSpace, Configuration

from openbox.utils.history import Observation
from openbox.experimental.online.base_online_advisor import OnlineAdvisor


class RandomSearch(OnlineAdvisor):

    def is_converged(self):
        return False

    def __init__(self,
                 config_space: ConfigurationSpace,
                 x0: Configuration=None,
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,
                 ):
        super().__init__(config_space=config_space, x0=x0, batch_size=batch_size, output_dir=output_dir,
                         task_id=task_id, random_state=random_state)
        self.dim = len(config_space.keys())

        self.type = "Global"
        self.config = None

    def get_suggestion(self):
        self.config = self.config_space.sample_configuration()
        return self.config

    def update_observation(self, observation: Observation):
        self.history.update_observation(observation)
