# License: MIT

import copy
import numpy as np

from openbox import logger
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.constants import SUCCESS
from openbox.core.generic_advisor import Advisor
from openbox.utils.history_container import Observation
from openbox.utils.util_funcs import deprecate_kwarg


class SyncBatchAdvisor(Advisor):
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
            history_bo_data=None,
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
        super().__init__(config_space,
                         num_objectives=num_objectives,
                         num_constraints=num_constraints,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         init_strategy=init_strategy,
                         history_bo_data=history_bo_data,
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

        assert self.batch_strategy in ['default', 'median_imputation', 'local_penalization', 'reoptimization']

        if self.num_objectives > 1 or self.num_constraints > 0:
            # local_penalization only supports single objective with no constraint
            assert self.batch_strategy in ['default', 'median_imputation', 'reoptimization']

        if self.batch_strategy == 'local_penalization':
            self.acq_type = 'lpei'

    def get_suggestions(self, batch_size=None, history_container=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size >= 1
        if history_container is None:
            history_container = self.history_container

        num_config_evaluated = len(history_container.configurations)
        num_config_successful = len(history_container.successful_perfs)

        if num_config_evaluated < self.init_num:
            if self.initial_configurations is not None:  # self.init_num equals to len(self.initial_configurations)
                next_configs = self.initial_configurations[num_config_evaluated: num_config_evaluated + batch_size]
                if len(next_configs) < batch_size:
                    next_configs.extend(
                        self.sample_random_configs(batch_size - len(next_configs), history_container))
                return next_configs
            else:
                return self.sample_random_configs(batch_size, history_container)

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(batch_size, history_container)

        if num_config_successful < max(self.init_num, 1):
            logger.warning('No enough successful initial trials! Sample random configurations.')
            return self.sample_random_configs(batch_size, history_container)

        X = convert_configurations_to_array(history_container.configurations)
        Y = history_container.get_transformed_perfs(transform=None)
        # cY = history_container.get_transformed_constraint_perfs(transform='bilog')

        batch_configs_list = list()

        if self.batch_strategy == 'median_imputation':
            # set bilog_transform=False to get real cY for estimating median
            cY = history_container.get_transformed_constraint_perfs(transform=None)

            estimated_y = np.median(Y, axis=0).reshape(-1).tolist()
            estimated_c = np.median(cY, axis=0).tolist() if self.num_constraints > 0 else None
            batch_history_container = copy.deepcopy(history_container)

            for batch_i in range(batch_size):
                # use super class get_suggestion
                curr_batch_config = super().get_suggestion(batch_history_container)

                # imputation
                observation = Observation(config=curr_batch_config, objectives=estimated_y, constraints=estimated_c,
                                          trial_state=SUCCESS, elapsed_time=None, extra_info=None)
                batch_history_container.update_observation(observation)
                batch_configs_list.append(curr_batch_config)

        elif self.batch_strategy == 'local_penalization':
            # local_penalization only supports single objective with no constraint
            self.surrogate_model.train(X, Y)
            incumbent_value = history_container.get_incumbents()[0][1]
            # L = self.estimate_L(X)
            for i in range(batch_size):
                if self.rng.random() < self.rand_prob:
                    # sample random configuration proportionally
                    logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
                    cur_config = self.sample_random_configs(1, history_container,
                                                            excluded_configs=batch_configs_list)[0]
                else:
                    self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                                     num_data=len(history_container.data),
                                                     batch_configs=batch_configs_list)

                    challengers = self.optimizer.maximize(
                        runhistory=history_container,
                        num_points=5000,
                    )
                    cur_config = challengers.challengers[0]
                batch_configs_list.append(cur_config)
        elif self.batch_strategy == 'reoptimization':
            surrogate_trained = False
            for i in range(batch_size):
                if self.rng.random() < self.rand_prob:
                    # sample random configuration proportionally
                    logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
                    cur_config = self.sample_random_configs(1, history_container,
                                                            excluded_configs=batch_configs_list)[0]
                else:
                    if not surrogate_trained:
                        # set return_list=True to ensure surrogate trained
                        candidates = super().get_suggestion(history_container, return_list=True)
                        surrogate_trained = True
                    else:
                        # re-optimize acquisition function
                        challengers = self.optimizer.maximize(runhistory=history_container,
                                                              num_points=5000)
                        candidates = challengers.challengers
                    cur_config = None
                    for config in candidates:
                        if config not in batch_configs_list and config not in history_container.configurations:
                            cur_config = config
                            break
                    if cur_config is None:
                        logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                                            'Sample random config.' % (len(candidates),))
                        cur_config = self.sample_random_configs(1, history_container,
                                                                excluded_configs=batch_configs_list)[0]
                batch_configs_list.append(cur_config)
        elif self.batch_strategy == 'default':
            # select first N candidates
            candidates = super().get_suggestion(history_container, return_list=True)
            idx = 0
            while len(batch_configs_list) < batch_size:
                if idx >= len(candidates):
                    logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                                        'Sample random config.' % (len(candidates),))
                    cur_config = self.sample_random_configs(1, history_container,
                                                            excluded_configs=batch_configs_list)[0]
                elif self.rng.random() < self.rand_prob:
                    # sample random configuration proportionally
                    logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
                    cur_config = self.sample_random_configs(1, history_container,
                                                            excluded_configs=batch_configs_list)[0]
                else:
                    cur_config = None
                    while idx < len(candidates):
                        conf = candidates[idx]
                        idx += 1
                        if conf not in batch_configs_list and conf not in history_container.configurations:
                            cur_config = conf
                            break
                if cur_config is not None:
                    batch_configs_list.append(cur_config)

        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.batch_strategy)
        return batch_configs_list
