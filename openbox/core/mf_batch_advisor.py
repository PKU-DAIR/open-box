# License: MIT

import copy
import numpy as np

from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.core.advisor import Advisor


class MFBatchAdvisor(Advisor):
    def __init__(self, config_space,
                 initial_trials=3,
                 initial_configurations=None,
                 optimization_strategy='bo',
                 batch_strategy='median_imputation',
                 history_bo_data=None,
                 surrogate_type='mfgpe',
                 output_dir='logs',
                 task_id='default_task_id',
                 rng=None):

        self.batch_strategy = batch_strategy
        super().__init__(config_space,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         optimization_strategy=optimization_strategy,
                         history_bo_data=history_bo_data,
                         surrogate_type=surrogate_type,
                         output_dir=output_dir,
                         task_id=task_id,
                         rng=rng)

        if batch_strategy == 'median_imputation':
            acq_type = 'ei'
        elif batch_strategy == 'local_penalization':
            acq_type = 'lpei'
        else:
            raise ValueError('Unsupported batch strategy - %s.' % batch_strategy)
        super(MFBatchAdvisor, self).setup_bo_basics(acq_type=acq_type)

    def get_suggestions(self, n_suggestions):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        num_failed_trial = len(self.failed_configurations)
        failed_perfs = list() if self.max_y is None else [self.max_y] * num_failed_trial
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        num_config_evaluated = len(self.perfs)
        batch_configs_list = list()

        if num_config_evaluated < self.init_num or self.optimization_strategy == 'random':
            return self.sample_random_configs(n_suggestions)

        if self.batch_strategy == 'median_imputation':
            estimated_y = np.median(Y)
            batch_history_container = copy.deepcopy(self.history_container)
            for i in range(n_suggestions):
                self.surrogate_model.train(X, Y, snapshot=(i == 0))
                incumbent_value = batch_history_container.get_incumbents()[0][1]
                self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                                 num_data=len(batch_history_container.data))

                challengers = self.optimizer.maximize(
                    runhistory=batch_history_container,
                    num_points=5000
                )

                is_repeated_config = True
                repeated_time = 0
                curr_batch_config = None

                while is_repeated_config:
                    try:
                        curr_batch_config = challengers.challengers[repeated_time]
                        batch_history_container.add(curr_batch_config, estimated_y)
                    except ValueError:
                        is_repeated_config = True
                        repeated_time += 1
                    else:
                        is_repeated_config = False

                batch_configs_list.append(curr_batch_config)
                X = np.append(X, convert_configurations_to_array([curr_batch_config]), axis=0)
                Y = np.append(Y, estimated_y)

        elif self.batch_strategy == 'local_penalization':
            self.surrogate_model.train(X, Y)
            incumbent_value = self.history_container.get_incumbents()[0][1]
            # L = self.estimate_L(X)
            for i in range(n_suggestions):
                self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                                 num_data=len(self.history_container.data),
                                                 batch_configs=batch_configs_list)

                challengers = self.optimizer.maximize(
                    runhistory=self.history_container,
                    num_points=5000
                )
                batch_configs_list.append(challengers.challengers[0])
        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.batch_strategy)
        return batch_configs_list

    # TODO: Add state input
    def update_mf_observations(self, mf_observations):
        self.surrogate_model.update_mf_trials(mf_observations)
        for hpo_evaluation_data in mf_observations[-1:]:
            for _config, _config_perf in hpo_evaluation_data.items():
                if _config not in self.configurations:
                    self.configurations.append(_config)
                    self.perfs.append(_config_perf)
                    self.history_container.add(_config, _config_perf)
