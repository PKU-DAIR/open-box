# License: MIT

from openbox.utils.history import Observation
from openbox.core.generic_advisor import Advisor
from openbox.utils.util_funcs import deprecate_kwarg


class RandomAdvisor(Advisor):
    """
    Random Advisor Class, which adopts the random policy to sample a configuration.
    """

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space,
                 num_objectives=1,
                 num_constraints=0,
                 initial_trials=3,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 transfer_learning_history=None,
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 ref_point=None,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,
                 logger_kwargs: dict = None,
                 **kwargs):

        super().__init__(
            config_space=config_space, num_objectives=num_objectives, num_constraints=num_constraints,
            initial_trials=initial_trials, initial_configurations=initial_configurations,
            init_strategy=init_strategy, transfer_learning_history=transfer_learning_history,
            rand_prob=1, optimization_strategy='random',
            surrogate_type=surrogate_type, acq_type=acq_type, acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point, output_dir=output_dir, task_id=task_id, random_state=random_state,
            logger_kwargs=logger_kwargs, **kwargs,
        )   # todo: do not derive from BO advisor

    def get_suggestion(self, history=None):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history is None:
            history = self.history
        return self.sample_random_configs(1, history)[0]

    def update_observation(self, observation: Observation):
        return self.history.update_observation(observation)

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
        assert isinstance(self.num_objectives, int) and self.num_objectives >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

    def setup_bo_basics(self):
        return
