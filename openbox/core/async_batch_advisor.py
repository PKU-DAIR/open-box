# License: MIT

import copy
import numpy as np

from openbox import logger
from openbox.utils.constants import SUCCESS
from openbox.utils.history import Observation
from openbox.core.generic_advisor import Advisor
from openbox.utils.util_funcs import deprecate_kwarg


class AsyncBatchAdvisor(Advisor):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            batch_size=4,
            batch_strategy='default',
            initial_trials=3,
            initial_configurations=None,
            init_strategy='random_explore_first',
            transfer_learning_history=None,
            rand_prob=0.1,
            optimization_strategy='bo',
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            ref_point=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
    ):

        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
        self.running_configs = list()
        self.bo_start_n = 3
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

    def check_setup(self):
        super().check_setup()

        if self.batch_strategy is None:
            self.batch_strategy = 'default'

        assert self.batch_strategy in ['default', 'median_imputation', 'local_penalization']

        if self.num_objectives > 1 or self.num_constraints > 0:
            # local_penalization only supports single objective with no constraint
            assert self.batch_strategy in ['default', 'median_imputation', ]

        if self.batch_strategy == 'local_penalization':
            self.acq_type = 'lpei'

    def get_suggestion(self, history=None):
        logger.info('#Call get_suggestion. len of running configs = %d.' % len(self.running_configs))
        config = self._get_suggestion(history)
        self.running_configs.append(config)
        return config

    def _get_suggestion(self, history=None):
        if history is None:
            history = self.history

        num_config_all = len(history) + len(self.running_configs)
        num_config_successful = history.get_success_count()

        if (num_config_all < self.init_num) or \
                num_config_successful < self.bo_start_n or \
                self.optimization_strategy == 'random':
            if num_config_all >= len(self.initial_configurations):
                _config = self.sample_random_configs(1, history)[0]
            else:
                _config = self.initial_configurations[num_config_all]
            return _config

        # sample random configuration proportionally
        if self.rng.random() < self.rand_prob:
            logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
            return self.sample_random_configs(1, history,
                                              excluded_configs=self.running_configs)[0]

        X = history.get_config_array(transform='scale')
        Y = history.get_objectives(transform='infeasible')
        # cY = history.get_constraints(transform='bilog')

        if self.batch_strategy == 'median_imputation':
            # get real cY for estimating median. do not use bilog transform.
            cY = history.get_constraints(transform='failed')

            estimated_y = np.median(Y, axis=0).reshape(-1).tolist()
            estimated_c = np.median(cY, axis=0).tolist() if self.num_constraints > 0 else None
            batch_history = copy.deepcopy(history)
            # imputation
            for config in self.running_configs:
                observation = Observation(config=config, objectives=estimated_y, constraints=estimated_c,
                                          trial_state=SUCCESS, elapsed_time=None, extra_info=None)
                batch_history.update_observation(observation)

            # use super class get_suggestion
            return super().get_suggestion(batch_history)

        elif self.batch_strategy == 'local_penalization':
            # local_penalization only supports single objective with no constraint
            self.surrogate_model.train(X, Y)
            incumbent_value = history.get_incumbent_value()
            self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                             num_data=len(history),
                                             batch_configs=self.running_configs)

            challengers = self.optimizer.maximize(
                runhistory=history,
                num_points=5000
            )
            return challengers.challengers[0]

        elif self.batch_strategy == 'default':
            # select first N candidates
            candidates = super().get_suggestion(history, return_list=True)

            for config in candidates:
                if config not in self.running_configs and config not in history.configurations:
                    return config

            logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                                'Sample random config.' % (len(candidates),))
            return self.sample_random_configs(1, history,
                                              excluded_configs=self.running_configs)[0]
        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.batch_strategy)

    def update_observation(self, observation: Observation):
        config = observation.config
        assert config in self.running_configs
        self.running_configs.remove(config)
        super().update_observation(observation)
