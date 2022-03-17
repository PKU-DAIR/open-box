# License: MIT

from openbox.core.base import Observation
from openbox.core.generic_advisor import Advisor


class RandomAdvisor(Advisor):
    """
    Random Advisor Class, which adopts the random policy to sample a configuration.
    """

    def __init__(self, config_space,
                 num_objs=1,
                 num_constraints=0,
                 initial_trials=3,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data=None,
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 ref_point=None,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None,
                 **kwargs):

        super().__init__(
            config_space=config_space, num_objs=num_objs, num_constraints=num_constraints,
            initial_trials=initial_trials, initial_configurations=initial_configurations,
            init_strategy=init_strategy, history_bo_data=history_bo_data,
            rand_prob=1, optimization_strategy='random',
            surrogate_type=surrogate_type, acq_type=acq_type, acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point, output_dir=output_dir, task_id=task_id, random_state=random_state,
            **kwargs,
        )   # todo: do not derive from BO advisor

    def get_suggestion(self, history_container=None):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history_container is None:
            history_container = self.history_container
        return self.sample_random_configs(1, history_container)[0]

    def update_observation(self, observation: Observation):
        return self.history_container.update_observation(observation)

    def algo_auto_selection(self):
        return

    def check_setup(self):
        """
        Check optimization_strategy
        Returns
        -------
        None
        """
        assert self.optimization_strategy in ['random']
        assert isinstance(self.num_objs, int) and self.num_objs >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

    def setup_bo_basics(self):
        return
