# License: MIT

import sys
import traceback
import copy
import numpy as np
from openbox.acquisition_function.acquisition import EI, LPEI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb
from openbox.utils.util_funcs import get_types, get_rng
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.facade.bo_facade import BaseFacade


class BatchBayesianOptimization(BaseFacade):
    def __init__(self, objective_function, config_space,
                 sample_strategy='bo',
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 initial_batch=1,
                 batch_size=3,
                 task_id='default_task_id',
                 rng=None):
        super().__init__(config_space, task_id, output_dir=logging_dir)
        self.logger = super()._get_logger(self.__class__.__name__)
        if rng is None:
            run_id, rng = get_rng()

        self.batch_size = batch_size
        self.init_batch = initial_batch
        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        self.time_limit_per_trial = time_limit_per_trial
        self.default_obj_value = MAXINT
        self.sample_strategy = sample_strategy

        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()

        # Initialize the basic component in BO.
        self.config_space.seed(rng.randint(MAXINT))
        self.objective_function = objective_function
        types, bounds = get_types(config_space)
        # TODO: what is the feature array.
        self.model = RandomForestWithInstances(types=types, bounds=bounds, seed=rng.randint(MAXINT))
        if self.sample_strategy == 'local_penalization':
            self.acquisition_function = LPEI(self.model)
        else:
            self.acquisition_function = EI(self.model)
        self.optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.acquisition_function,
            config_space=self.config_space,
            rng=np.random.RandomState(seed=rng.randint(MAXINT)),
            max_steps=self.sls_max_steps,
            n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
            n_sls_iterations=self.n_sls_iterations
        )
        self._random_search = RandomSearch(
            self.acquisition_function, self.config_space, rng
        )
        self.random_configuration_chooser = ChooserProb(prob=0.25, rng=rng)

    def run(self):
        while self.iteration_id < self.max_iterations:
            self.iterate()

    def iterate(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        failed_perfs = list() if self.max_y is None else [self.max_y] * len(self.failed_configurations)
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        config_list = self.choose_next(X, Y)
        trial_state_list = list()
        trial_info_list = list()
        perf_list = list()

        for i, config in enumerate(config_list):
            trial_state_list.append(SUCCESS)
            trial_info_list.append(None)
            perf_list.append(None)

            if config not in (self.configurations + self.failed_configurations):
                # Evaluate this configuration.
                try:
                    args, kwargs = (config,), dict()
                    timeout_status, _result = time_limit(self.objective_function, self.time_limit_per_trial,
                                                         args=args, kwargs=kwargs)
                    if timeout_status:
                        raise TimeoutException(
                            'Timeout: time limit for this evaluation is %.1fs' % self.time_limit_per_trial)
                    else:
                        perf_list[i] = _result
                except Exception as e:
                    if isinstance(e, TimeoutException):
                        trial_state_list[i] = TIMEOUT
                    else:
                        traceback.print_exc(file=sys.stdout)
                        trial_state_list[i] = FAILED
                    perf_list[i] = MAXINT
                    trial_info_list[i] = str(e)
                    self.logger.error(trial_info_list[i])

                if trial_state_list[i] == SUCCESS and perf_list[i] < MAXINT:
                    if len(self.configurations) == 0:
                        self.default_obj_value = perf_list[i]

                    self.configurations.append(config)
                    self.perfs.append(perf_list[i])
                    self.history_container.add(config, perf_list[i])

                    self.perc = np.percentile(self.perfs, self.scale_perc)
                    self.min_y = np.min(self.perfs)
                    self.max_y = np.max(self.perfs)
                else:
                    self.failed_configurations.append(config)
            else:
                self.logger.debug('This configuration has been evaluated! Skip it.')
                if config in self.configurations:
                    config_idx = self.configurations.index(config)
                    trial_state_list[i], perf_list[i] = SUCCESS, self.perfs[config_idx]
                else:
                    trial_state_list[i], perf_list[i] = FAILED, MAXINT

        self.iteration_id += 1
        self.logger.info(
            'Iteration-%d, objective improvement: %.4f' % (
                self.iteration_id, max(0, self.default_obj_value - min(perf_list))))
        return config_list, trial_state_list, perf_list, trial_info_list

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        # Select a batch of configs to evaluate next.
        _config_num = X.shape[0]
        batch_configs_list = list()

        if _config_num < self.init_batch * self.batch_size or self.sample_strategy == 'random':
            for i in range(self.batch_size):
                batch_configs_list.append(self.sample_config())
            return batch_configs_list

        if self.sample_strategy == 'median_imputation':
            estimated_y = np.mean(Y)
            batch_history_container = copy.deepcopy(self.history_container)
            for i in range(self.batch_size):
                self.model.train(X, Y)

                incumbent_value = batch_history_container.get_incumbents()[0][1]

                self.acquisition_function.update(model=self.model, eta=incumbent_value,
                                                 num_data=len(batch_history_container.data))

                challengers = self.optimizer.maximize(
                    runhistory=batch_history_container,
                    num_points=5000,
                    random_configuration_chooser=self.random_configuration_chooser
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
                X = np.append(X, curr_batch_config.get_array().reshape(1, -1), axis=0)
                Y = np.append(Y, estimated_y)
                estimated_y = np.mean(Y)

        elif self.sample_strategy == 'local_penalization':
            self.model.train(X, Y)
            incumbent_value = self.history_container.get_incumbents()[0][1]
            # L = self.estimate_L(X)
            for i in range(self.batch_size):
                self.acquisition_function.update(model=self.model, eta=incumbent_value,
                                                 num_data=len(self.history_container.data),
                                                 batch_configs=batch_configs_list)

                challengers = self.optimizer.maximize(
                    runhistory=self.history_container,
                    num_points=5000,
                    random_configuration_chooser=self.random_configuration_chooser
                )
                batch_configs_list.append(challengers.challengers[0])
        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.sample_strategy)

        return batch_configs_list

    def sample_config(self):
        config = None
        _sample_cnt, _sample_limit = 0, 10000
        while True:
            _sample_cnt += 1
            config = self.config_space.sample_configuration()
            if config not in (self.configurations + self.failed_configurations):
                break
            if _sample_cnt >= _sample_limit:
                config = self.config_space.sample_configuration()
                break
        return config

    # def estimate_L(self, X):  # X: N*D
    #     """
    #         Estimate the Lipschitz constant of f by taking maximum norm of the gradient of f.
    #         Calculate numerical gradient = f(x+h) - f(x-h) / 2h , where h is very small
    #     """
    #     eps = 1e-3
    #     for i in range(10000):
    #         X = np.vstack((X, self.sample_config().get_array()))
    #
    #     # print("=" * 20, "X", X)
    #     dX = np.zeros_like(X)
    #     for j in range(X.shape[1]):
    #         h = np.zeros_like(X)
    #         h[:, j] = eps
    #         # print("=" * 20, "X+h", X + h)
    #         f_pos, _ = self.surrogate.predict_marginalized_over_instances(X + h)
    #         f_neg, _, = self.surrogate.predict_marginalized_over_instances(X - h)
    #         # print("=" * 20, "f_pos", dX)
    #         # print("=" * 20, "f_neg", dX)
    #         f_pos = f_pos.reshape(-1)
    #         f_neg = f_neg.reshape(-1)
    #         dX[:, j] = (f_pos - f_neg) / (2 * eps)
    #     # print("="*20,"dX",dX)
    #
    #     norm_dX = np.linalg.norm(dX, axis=1)  # shape=(N,)
    #
    #     return np.max(norm_dX)
