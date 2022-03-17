# License: MIT

import os
import time
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from openbox.apps.multi_fidelity.async_mq_hb import async_mqHyperband
from openbox.apps.multi_fidelity.utils import RUNNING, COMPLETED, PROMOTED
from openbox.apps.multi_fidelity.utils import sample_configuration
from openbox.apps.multi_fidelity.utils import minmax_normalization, std_normalization
from openbox.surrogate.base.rf_ensemble import RandomForestEnsemble

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer


class async_mqMFES(async_mqHyperband):
    """
    The implementation of Asynchronous MFES (combine ASHA and MFES)
    """

    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 rand_prob=0.3,
                 use_bohb=False,
                 init_weight=None, update_enable=True,
                 weight_method='rank_loss_p_norm',
                 fusion_method='idp',
                 power_num=3,
                 random_state=1,
                 method_id='mqAsyncMFES',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 seed=1,
                 ip='',
                 port=13579,
                 authkey=b'abc'):
        super().__init__(objective_func, config_space, R, eta=eta, skip_outer_loop=skip_outer_loop,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)
        self.seed = seed
        self.last_n_iteration = None
        self.use_bohb_strategy = use_bohb
        self.update_enable = update_enable
        self.fusion_method = fusion_method
        # Parameter for weight method `rank_loss_p_norm`.
        self.power_num = power_num
        # Specify the weight learning method.
        self.weight_method = weight_method
        self.weight_update_id = 0
        self.weight_changed_cnt = 0

        if init_weight is None:
            init_weight = [1. / self.s_max] * self.s_max + [0.]
            assert len(init_weight) == (self.s_max + 1)
        self.logger.info("Initialize weight to %s" % init_weight[:self.s_max + 1])
        types, bounds = get_types(config_space)

        if not self.use_bohb_strategy:
            self.surrogate = RandomForestEnsemble(types, bounds, self.s_max, self.eta,
                                                  init_weight, self.fusion_method)
        else:
            self.surrogate = RandomForestWithInstances(types, bounds, seed=self.seed)
        self.acquisition_function = EI(model=self.surrogate)

        self.iterate_id = 0
        self.iterate_r = list()
        self.hist_weights = list()

        # Saving evaluation statistics in Hyperband.
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = list()
            self.target_y[r] = list()

        # BO optimizer settings.
        self.history_container = HistoryContainer(task_id=self.method_name)
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        self.rng = np.random.RandomState(seed=self.seed)
        self.acq_optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.acquisition_function,
            config_space=self.config_space,
            rng=self.rng,
            max_steps=self.sls_max_steps,
            n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
            n_sls_iterations=self.n_sls_iterations,
            rand_prob=0.0,
        )
        self.random_configuration_chooser = ChooserProb(prob=rand_prob, rng=self.rng)
        self.random_check_idx = 0

    def update_observation(self, config, perf, n_iteration):
        rung_id = self.get_rung_id(self.bracket, n_iteration)

        updated = False
        for job in self.bracket[rung_id]['jobs']:
            _job_status, _config, _perf, _extra_conf = job
            if _config == config:
                assert _job_status == RUNNING
                job[0] = COMPLETED
                job[2] = perf
                updated = True
                break
        assert updated
        # print('=== bracket after update_observation:', self.get_bracket_status(self.bracket))

        configs_running = list()
        for _config in self.bracket[rung_id]['configs']:
            if _config not in self.target_x[n_iteration]:
                configs_running.append(_config)
        value_imputed = np.median(self.target_y[n_iteration])

        n_iteration = int(n_iteration)
        self.target_x[n_iteration].append(config)
        self.target_y[n_iteration].append(perf)

        if n_iteration == self.R:
            self.incumbent_configs.append(config)
            self.incumbent_perfs.append(perf)
            # Update history container.
            self.history_container.add(config, perf)

        # Refit the ensemble surrogate model.
        configs_train = self.target_x[n_iteration] + configs_running
        results_train = self.target_y[n_iteration] + [value_imputed] * len(configs_running)
        results_train = np.array(std_normalization(results_train), dtype=np.float64)
        if not self.use_bohb_strategy:
            self.surrogate.train(convert_configurations_to_array(configs_train), results_train, r=n_iteration)
        else:
            if n_iteration == self.R:
                self.surrogate.train(convert_configurations_to_array(configs_train), results_train)

    def choose_next(self):
        """
        sample a config according to MFES. give iterations according to Hyperband strategy.
        """
        next_config = None
        next_n_iteration = self.get_next_n_iteration()
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        # Update weight when the inner loop of hyperband is finished
        if self.last_n_iteration != next_n_iteration and not self.use_bohb_strategy:
            if self.update_enable and self.weight_update_id > self.s_max:
                self.update_weight()
            self.weight_update_id += 1
        self.last_n_iteration = next_n_iteration

        # sample config
        excluded_configs = self.bracket[next_rung_id]['configs']
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
        else:
            # Like BOHB, sample a fixed percentage of random configurations.
            self.random_check_idx += 1
            if self.random_configuration_chooser.check(self.random_check_idx):
                next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)
            else:
                acq_configs = self.get_bo_candidates()
                for config in acq_configs:
                    if config not in self.bracket[next_rung_id]['configs']:
                        next_config = config
                        break
                if next_config is None:
                    self.logger.warning('Cannot get a non duplicate configuration from bo candidates. '
                                        'Sample a random one.')
                    next_config = sample_configuration(self.config_space, excluded_configs=excluded_configs)

        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_bo_candidates(self):
        std_incumbent_value = np.min(std_normalization(self.target_y[self.iterate_r[-1]]))
        # Update surrogate model in acquisition function.
        self.acquisition_function.update(model=self.surrogate, eta=std_incumbent_value,
                                         num_data=len(self.incumbent_configs))

        challengers = self.acq_optimizer.maximize(
            runhistory=self.history_container,
            num_points=5000,
        )
        return challengers.challengers

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if bool(y_true[idx] > y_true[inner_idx]) == bool(y_pred[idx] > y_pred[inner_idx]):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def update_weight(self):
        start_time = time.time()

        max_r = self.iterate_r[-1]
        incumbent_configs = self.target_x[max_r]
        test_x = convert_configurations_to_array(incumbent_configs)
        test_y = np.array(self.target_y[max_r], dtype=np.float64)

        r_list = self.surrogate.surrogate_r
        K = len(r_list)

        if len(test_y) >= 3:
            # Get previous weights
            if self.weight_method == 'rank_loss_p_norm':
                preserving_order_p = list()
                preserving_order_nums = list()
                for i, r in enumerate(r_list):
                    fold_num = 5
                    if i != K - 1:
                        mean, var = self.surrogate.surrogate_container[r].predict(test_x)
                        tmp_y = np.reshape(mean, -1)
                        preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
                        preserving_order_p.append(preorder_num / pair_num)
                        preserving_order_nums.append(preorder_num)
                    else:
                        if len(test_y) < 2 * fold_num:
                            preserving_order_p.append(0)
                        else:
                            # 5-fold cross validation.
                            kfold = KFold(n_splits=fold_num)
                            cv_pred = np.array([0] * len(test_y))
                            for train_idx, valid_idx in kfold.split(test_x):
                                train_configs, train_y = test_x[train_idx], test_y[train_idx]
                                valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                                types, bounds = get_types(self.config_space)
                                _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                                _surrogate.train(train_configs, train_y)
                                pred, _ = _surrogate.predict(valid_configs)
                                cv_pred[valid_idx] = pred.reshape(-1)
                            preorder_num, pair_num = self.calculate_preserving_order_num(cv_pred, test_y)
                            preserving_order_p.append(preorder_num / pair_num)
                            preserving_order_nums.append(preorder_num)

                trans_order_weight = np.array(preserving_order_p)
                power_sum = np.sum(np.power(trans_order_weight, self.power_num))
                new_weights = np.power(trans_order_weight, self.power_num) / power_sum

            elif self.weight_method == 'rank_loss_prob':
                # For basic surrogate i=1:K-1.
                mean_list, var_list = list(), list()
                for i, r in enumerate(r_list[:-1]):
                    mean, var = self.surrogate.surrogate_container[r].predict(test_x)
                    mean_list.append(np.reshape(mean, -1))
                    var_list.append(np.reshape(var, -1))
                sample_num = 100
                min_probability_array = [0] * K
                for _ in range(sample_num):
                    order_preseving_nums = list()

                    # For basic surrogate i=1:K-1.
                    for idx in range(K - 1):
                        sampled_y = self.rng.normal(mean_list[idx], var_list[idx])
                        _num, _ = self.calculate_preserving_order_num(sampled_y, test_y)
                        order_preseving_nums.append(_num)

                    fold_num = 5
                    # For basic surrogate i=K. cv
                    if len(test_y) < 2 * fold_num:
                        order_preseving_nums.append(0)
                    else:
                        # 5-fold cross validation.
                        kfold = KFold(n_splits=fold_num)
                        cv_pred = np.array([0] * len(test_y))
                        for train_idx, valid_idx in kfold.split(test_x):
                            train_configs, train_y = test_x[train_idx], test_y[train_idx]
                            valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                            types, bounds = get_types(self.config_space)
                            _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                            _surrogate.train(train_configs, train_y)
                            _pred, _var = _surrogate.predict(valid_configs)
                            sampled_pred = self.rng.normal(_pred.reshape(-1), _var.reshape(-1))
                            cv_pred[valid_idx] = sampled_pred
                        _num, _ = self.calculate_preserving_order_num(cv_pred, test_y)
                        order_preseving_nums.append(_num)
                    max_id = np.argmax(order_preseving_nums)
                    min_probability_array[max_id] += 1
                new_weights = np.array(min_probability_array) / sample_num
            else:
                raise ValueError('Invalid weight method: %s!' % self.weight_method)
        else:
            old_weights = list()
            for i, r in enumerate(r_list):
                _weight = self.surrogate.surrogate_weight[r]
                old_weights.append(_weight)
            new_weights = old_weights.copy()

        self.logger.info('[%s] %d-th Updating weights: %s' % (
            self.weight_method, self.weight_changed_cnt, str(new_weights)))

        # Assign the weight to each basic surrogate.
        for i, r in enumerate(r_list):
            self.surrogate.surrogate_weight[r] = new_weights[i]
        self.weight_changed_cnt += 1
        # Save the weight data.
        self.hist_weights.append(new_weights)
        dir_path = os.path.join(self.data_directory, 'saved_weights')
        file_name = 'mfes_weights_%s.npy' % (self.method_name,)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(os.path.join(dir_path, file_name), np.asarray(self.hist_weights))
        self.logger.info('update_weight() cost %.2fs. new weights are saved to %s'
                         % (time.time() - start_time, os.path.join(dir_path, file_name)))

    def get_weights(self):
        return self.hist_weights

