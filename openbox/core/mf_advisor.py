# License: MIT

from openbox.core.generic_advisor import Advisor
from openbox.utils.history import Observation, History
from openbox.utils.util_funcs import deprecate_kwarg


class MFAdvisor(Advisor):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            initial_trials=3,
            initial_configurations=None,
            init_strategy='random_explore_first',
            transfer_learning_history=None,
            warm_start_strategy='topk',
            warm_start_num=None,
            rand_prob=0.1,
            optimization_strategy='bo',
            surrogate_type='mfgpe',
            acq_type='ei',
            acq_optimizer_type='local_random',
            ref_point=None,
            early_stop=False,
            early_stop_kwargs=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
            **kwargs,
    ):
        super().__init__(
            config_space=config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            initial_trials=initial_trials,
            initial_configurations=initial_configurations,
            init_strategy=init_strategy,
            transfer_learning_history=transfer_learning_history,
            warm_start_strategy=warm_start_strategy,
            warm_start_num=warm_start_num,
            rand_prob=rand_prob,
            optimization_strategy=optimization_strategy,
            surrogate_type=surrogate_type,
            acq_type=acq_type,
            acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point,
            early_stop=early_stop,
            early_stop_kwargs=early_stop_kwargs,
            output_dir=output_dir,
            task_id=task_id,
            random_state=random_state,
            logger_kwargs=logger_kwargs,
            **kwargs,
        )
        self.history_list = []
        self.resource_identifiers = []

    @staticmethod
    def _round_resource_ratio(resource_ratio):
        return round(float(resource_ratio), 5)

    def _get_or_create_history(self, resource_ratio):
        if resource_ratio not in self.resource_identifiers:
            self.resource_identifiers.append(resource_ratio)
            history = History(
                task_id=self.task_id,
                num_objectives=self.num_objectives,
                num_constraints=self.num_constraints,
                config_space=self.config_space,
                ref_point=self.ref_point,
            )
            self.history_list.append(history)
        idx = self.resource_identifiers.index(resource_ratio)
        return self.history_list[idx]

    def _prepare_mf_surrogate(self):
        self.surrogate_model.update_mf_trials(self.history_list)
        self.surrogate_model.build_source_surrogates()

    def get_suggestion(self, history=None):
        self._prepare_mf_surrogate()
        return super().get_suggestion(history=history)

    def get_suggestions(self, batch_size=1, history=None):
        self._prepare_mf_surrogate()
        return super().get_suggestions(batch_size=batch_size, history=history)

    def update_observation(self, observation: Observation, resource_ratio=1.0):
        resource_ratio = self._round_resource_ratio(resource_ratio)
        mf_history = self._get_or_create_history(resource_ratio)
        mf_history.update_observation(observation)
        if resource_ratio == self._round_resource_ratio(1.0):
            return super().update_observation(observation)
