# License: MIT

import time
import os
import numpy as np
from math import log, ceil
from sklearn.model_selection import KFold
from scipy.optimize import minimize

from openbox.apps.multi_fidelity.mq_base_facade import mqBaseFacade
from openbox.apps.multi_fidelity.utils import sample_configurations, expand_configurations
from openbox.apps.multi_fidelity.utils import minmax_normalization, std_normalization
from openbox.surrogate.base.weighted_rf_ensemble import WeightedRandomForestCluster

from openbox.utils.util_funcs import get_types
from openbox.utils.config_space import ConfigurationSpace
from openbox.acquisition_function.acquisition import EI
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from openbox.acq_maximizer.random_configuration_chooser import ChooserProb
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer


class mqMFES(mqBaseFacade):
    """
    MFES-HB: https://arxiv.org/abs/2012.03011
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 num_iter=10000,
                 rand_prob=0.3,
                 init_weight=None, update_enable=True,
                 weight_method='rank_loss_p_norm', fusion_method='idp',
                 power_num=3,
                 random_state=1,
                 method_id='mqMFES',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        max_queue_len = 3 * R  # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)

        self.R = R
        self.eta = eta
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.num_iter = num_iter

        self.update_enable = update_enable
        self.fusion_method = fusion_method
        # Parameter for weight method `rank_loss_p_norm`.
        self.power_num = power_num
        # Specify the weight learning method.
        self.weight_method = weight_method
        self.weight_update_id = 0
        self.weight_changed_cnt = 0

        if init_weight is None:
            init_weight = [0.]
            init_weight.extend([1. / self.s_max] * self.s_max)
        assert len(init_weight) == (self.s_max + 1)
        self.logger.info('Weight method & flag: %s-%s' % (self.weight_method, str(self.update_enable)))
        self.logger.info("Initial weight is: %s" % init_weight[:self.s_max + 1])
        types, bounds = get_types(config_space)

        self.weighted_surrogate = WeightedRandomForestCluster(
            types, bounds, self.s_max, self.eta, init_weight, self.fusion_method
        )
        self.acquisition_function = EI(model=self.weighted_surrogate)

        self.incumbent_configs = []
        self.incumbent_perfs = []

        self.iterate_id = 0
        self.iterate_r = []
        self.hist_weights = list()

        # Saving evaluation statistics in Hyperband.
        self.target_x = dict()
        self.target_y = dict()
        for index, item in enumerate(np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
            r = int(item)
            self.iterate_r.append(r)
            self.target_x[r] = []
            self.target_y[r] = []

        # BO optimizer settings.
        self.configs = list()
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

    def iterate(self, skip_last=0):

        for s in reversed(range(self.s_max + 1)):

            if self.update_enable and self.weight_update_id > self.s_max:
                self.update_weight()
            self.weight_update_id += 1

            # Set initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # initial number of iterations per config
            r = int(self.R * self.eta ** (-s))

            # Choose a batch of configurations in different mechanisms.
            start_time = time.time()
            T = self.choose_next(n)
            time_elapsed = time.time() - start_time
            self.logger.info("[%s] Choosing next configurations took %.2f sec." % (self.method_name, time_elapsed))

            extra_info = None
            last_run_num = None

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iteration = r * self.eta ** (i)

                n_iter = n_iteration
                if last_run_num is not None and not self.restart_needed:
                    n_iter -= last_run_num
                last_run_num = n_iteration

                self.logger.info("%s: %d configurations x %d iterations each" %
                                 (self.method_name, int(n_configs), int(n_iteration)))

                ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)
                val_losses = [item['loss'] for item in ret_val]
                ref_list = [item['ref_id'] for item in ret_val]

                self.target_x[int(n_iteration)].extend(T)
                self.target_y[int(n_iteration)].extend(val_losses)

                if int(n_iteration) == self.R:
                    self.incumbent_configs.extend(T)
                    self.incumbent_perfs.extend(val_losses)
                    # Update history container.
                    for _config, _perf in zip(T, val_losses):
                        self.history_container.add(_config, _perf)

                # Select a number of best configurations for the next loop.
                # Filter out early stops, if any.
                indices = np.argsort(val_losses)
                if len(T) == sum(early_stops):
                    break
                if len(T) >= self.eta:
                    indices = [i for i in indices if not early_stops[i]]
                    T = [T[i] for i in indices]
                    extra_info = [ref_list[i] for i in indices]
                    reduced_num = int(n_configs / self.eta)
                    T = T[0:reduced_num]
                    extra_info = extra_info[0:reduced_num]
                else:
                    T = [T[indices[0]]]     # todo: confirm no filter early stops?
                    extra_info = [ref_list[indices[0]]]
                val_losses = [val_losses[i] for i in indices][0:len(T)]  # update: sorted
                incumbent_loss = val_losses[0]
                self.add_stage_history(self.stage_id, min(self.global_incumbent, incumbent_loss))
                self.stage_id += 1
            # self.remove_immediate_model()

            for item in self.iterate_r[self.iterate_r.index(r):]:
                # NORMALIZE Objective value: normalization
                normalized_y = std_normalization(self.target_y[item])
                self.weighted_surrogate.train(convert_configurations_to_array(self.target_x[item]),
                                              np.array(normalized_y, dtype=np.float64), r=item)

    def run(self, skip_last=0):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("%s algorithm: %d/%d iteration starts" % (self.method_name, iter, self.num_iter))
                start_time = time.time()
                self.iterate(skip_last=skip_last)
                time_elapsed = (time.time() - start_time) / 60
                self.logger.info("%d/%d-Iteration took %.2f min." % (iter, self.num_iter, time_elapsed))
                self.iterate_id += 1
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # Clean the immediate results.
            # self.remove_immediate_model()

    def get_bo_candidates(self, num_configs):
        # todo: parallel methods
        std_incumbent_value = np.min(std_normalization(self.target_y[self.iterate_r[-1]]))
        # Update surrogate model in acquisition function.
        self.acquisition_function.update(model=self.weighted_surrogate, eta=std_incumbent_value,
                                         num_data=len(self.history_container.data))

        challengers = self.acq_optimizer.maximize(
            runhistory=self.history_container,
            num_points=5000,
        )
        return challengers.challengers[:num_configs]

    def choose_next(self, num_config):
        if len(self.target_y[self.iterate_r[-1]]) == 0:
            configs = sample_configurations(self.config_space, num_config)
            self.configs.extend(configs)
            return configs

        config_candidates = list()
        acq_configs = self.get_bo_candidates(num_configs=2 * num_config)
        acq_idx = 0
        for idx in range(1, 1 + 2 * num_config):
            # Like BOHB, sample a fixed percentage of random configurations.
            if self.random_configuration_chooser.check(idx):
                _config = self.config_space.sample_configuration()
            else:
                _config = acq_configs[acq_idx]
                acq_idx += 1
            if _config not in config_candidates:
                config_candidates.append(_config)
            if len(config_candidates) >= num_config:
                break

        if len(config_candidates) < num_config:
            config_candidates = expand_configurations(config_candidates, self.config_space, num_config)

        _config_candidates = []
        for config in config_candidates:
            if config not in self.configs:  # Check if evaluated
                _config_candidates.append(config)
        self.configs.extend(_config_candidates)
        return _config_candidates

    @staticmethod
    def calculate_ranking_loss(y_pred, y_true):
        length = len(y_pred)
        y_pred = np.reshape(y_pred, -1)
        y_pred1 = np.tile(y_pred, (length, 1))
        y_pred2 = np.transpose(y_pred1)
        diff = y_pred1 - y_pred2
        y_true = np.reshape(y_true, -1)
        y_true1 = np.tile(y_true, (length, 1))
        y_true2 = np.transpose(y_true1)
        y_mask = (y_true1 - y_true2 > 0) + 0
        loss = np.sum(np.log(1 + np.exp(-diff)) * y_mask) / length
        return loss

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

        r_list = self.weighted_surrogate.surrogate_r
        K = len(r_list)

        if len(test_y) >= 3:
            # Get previous weights
            if self.weight_method == 'rank_loss_p_norm':
                preserving_order_p = list()
                preserving_order_nums = list()
                for i, r in enumerate(r_list):
                    fold_num = 5
                    if i != K - 1:
                        mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
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
                    mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
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
                _weight = self.weighted_surrogate.surrogate_weight[r]
                old_weights.append(_weight)
            new_weights = old_weights.copy()

        self.logger.info('[%s] %d-th Updating weights: %s' % (
            self.weight_method, self.weight_changed_cnt, str(new_weights)))

        # Assign the weight to each basic surrogate.
        for i, r in enumerate(r_list):
            self.weighted_surrogate.surrogate_weight[r] = new_weights[i]
        self.weight_changed_cnt += 1
        # Save the weight data.
        self.hist_weights.append(new_weights)
        dir_path = os.path.join(self.data_directory, 'saved_weights')
        file_name = 'mfes_weights_%s.npy' % (self.method_name,)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(os.path.join(dir_path, file_name), np.asarray(self.hist_weights))
        self.logger.info('update_weight() cost %.2fs. new weights are saved to %s'
                         % (time.time()-start_time, os.path.join(dir_path, file_name)))

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs

    def get_weights(self):
        return self.hist_weights
