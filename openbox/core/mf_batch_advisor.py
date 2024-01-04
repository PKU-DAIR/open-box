# License: MIT

from openbox import logger
from openbox.core.generic_advisor import Advisor
from openbox.utils.history import Observation, History
from openbox.utils.util_funcs import deprecate_kwarg


class MFBatchAdvisor(Advisor):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            batch_size=4,
            initial_trials=3,
            initial_configurations=None,
            init_strategy='random_explore_first',
            transfer_learning_history=None,
            rand_prob=0.1,
            optimization_strategy='bo',
            surrogate_type='mfgpe',
            acq_type='ei',
            acq_optimizer_type='local_random',
            ref_point=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
    ):

        self.batch_size = batch_size
        super().__init__(config_space,
                         num_objectives=num_objectives,
                         num_constraints=num_constraints,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         init_strategy=init_strategy,
                         transfer_learning_history=transfer_learning_history,
                         rand_prob=rand_prob,
                         optimization_strategy=optimization_strategy,
                         surrogate_type=surrogate_type,
                         acq_type=acq_type,
                         acq_optimizer_type=acq_optimizer_type,
                         ref_point=ref_point,
                         output_dir=output_dir,
                         task_id=task_id,
                         random_state=random_state,
                         logger_kwargs=logger_kwargs)
        self.history_list = list()
        self.resource_identifiers = list()

    def get_suggestions(self, batch_size=None, history=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size >= 1
        if history is None:
            history = self.history

        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()

        if num_config_evaluated < self.init_num:
            if self.initial_configurations is not None:  # self.init_num equals to len(self.initial_configurations)
                next_configs = self.initial_configurations[num_config_evaluated: num_config_evaluated + batch_size]
                if len(next_configs) < batch_size:
                    next_configs.extend(self.sample_random_configs(
                        self.config_space, batch_size - len(next_configs), excluded_configs=history.configurations))
                return next_configs
            else:
                return self.sample_random_configs(self.config_space, batch_size,
                                                  excluded_configs=history.configurations)

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(self.config_space, batch_size,
                                              excluded_configs=history.configurations)

        if num_config_successful < max(self.init_num, 1):
            logger.warning('No enough successful initial trials! Sample random configurations.')
            return self.sample_random_configs(self.config_space, batch_size,
                                              excluded_configs=history.configurations)

        batch_configs_list = list()

        # select first N candidates
        self.surrogate_model.update_mf_trials(self.history_list)
        self.surrogate_model.build_source_surrogates()
        candidates = super().get_suggestion(history, return_list=True)  # replace
        idx = 0
        while len(batch_configs_list) < batch_size:
            if idx >= len(candidates):
                logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                               'Sample random config.' % (len(candidates),))
                cur_config = self.sample_random_configs(
                    self.config_space, 1, excluded_configs=history.configurations + batch_configs_list)[0]
            elif self.rng.random() < self.rand_prob:
                # sample random configuration proportionally
                logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
                cur_config = self.sample_random_configs(
                    self.config_space, 1, excluded_configs=history.configurations + batch_configs_list)[0]
            else:
                cur_config = None
                while idx < len(candidates):
                    conf = candidates[idx]
                    idx += 1
                    if conf not in batch_configs_list and conf not in history.configurations:
                        cur_config = conf
                        break
            if cur_config is not None:
                batch_configs_list.append(cur_config)

        return batch_configs_list

    def update_observation(self, observation: Observation, resource_ratio):
        if resource_ratio not in self.resource_identifiers:
            self.resource_identifiers.append(resource_ratio)
            history = History(task_id=self.task_id, num_objectives=self.num_objectives,
                              num_constraints=self.num_constraints,
                              config_space=self.config_space,
                              ref_point=self.ref_point)
            self.history_list.append(history)

        self.history_list[self.resource_identifiers.index(resource_ratio)].update_observation(observation)

        if resource_ratio == 1:
            self.history.update_observation(observation)
