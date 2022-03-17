# License: MIT

import time
import numpy as np
from math import log, ceil
from openbox.apps.multi_fidelity.mq_base_facade import mqBaseFacade
from openbox.apps.multi_fidelity.utils import sample_configurations

from openbox.utils.config_space import ConfigurationSpace


class mqHyperband(mqBaseFacade):
    """ The implementation of Hyperband (HB).
        The paper can be found in http://www.jmlr.org/papers/volume18/16-558/16-558.pdf .
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 num_iter=10000,
                 random_state=1,
                 method_id='mqHyperband',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        max_queue_len = 3 * R   # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)

        self.num_iter = num_iter
        self.R = R      # Maximum iterations per configuration
        self.eta = eta  # Define configuration downsampling rate (default = 3)
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R

        self.incumbent_configs = list()
        self.incumbent_perfs = list()

    # This function can be called multiple times
    def iterate(self, skip_last=0):
        for s in reversed(range(self.s_max + 1)):
            # Initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # Initial number of iterations per config
            r = self.R * self.eta ** (-s)

            # Choose next n configurations.
            T = self.choose_next(n)
            incumbent_loss = np.inf
            extra_info = None
            last_run_num = None
            for i in range((s + 1) - int(skip_last)):  # Changed from s + 1
                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations.

                n_configs = n * self.eta ** (-i)
                n_iteration = r * self.eta ** (i)
                n_iter = n_iteration
                if last_run_num is not None and not self.restart_needed:
                    n_iter -= last_run_num
                last_run_num = n_iteration

                self.logger.info("%s: %d configurations x %d iterations each"
                                 % (self.method_name, int(n_configs), int(n_iteration)))

                ret_val, early_stops = self.run_in_parallel(T, n_iter, extra_info)
                val_losses = [item['loss'] for item in ret_val]
                ref_list = [item['ref_id'] for item in ret_val]

                self.update_incumbent_before_reduce(T, val_losses, n_iteration)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
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
                val_losses = [val_losses[i] for i in indices][0:len(T)]     # update: sorted
                incumbent_loss = val_losses[0]
                self.add_stage_history(self.stage_id, min(self.global_incumbent, incumbent_loss))
                self.stage_id += 1
            self.update_incumbent_after_reduce(T, incumbent_loss)
            # self.remove_immediate_model()

    def run(self, skip_last=0):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("%s algorithm: %d/%d iteration starts" % (self.method_name, iter, self.num_iter))
                start_time = time.time()
                self.iterate(skip_last=skip_last)
                time_elapsed = (time.time() - start_time) / 60
                self.logger.info("Iteration took %.2f min." % time_elapsed)
                self.save_intemediate_statistics()
            for i, obj in enumerate(self.incumbent_perfs):
                self.logger.info(
                    '%d-th config: %s, obj: %f.' % (i + 1, str(self.incumbent_configs[i]), self.incumbent_perfs[i]))
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # Clean the immediate results.
            # self.remove_immediate_model()

    def choose_next(self, num_config):
        # Sample n configurations uniformly.
        return sample_configurations(self.config_space, num_config)

    def update_incumbent_before_reduce(self, T, val_losses, n_iteration):
        return

    def update_incumbent_after_reduce(self, T, incumbent_loss):
        """
        update: T is sorted
        """
        if not np.isnan(incumbent_loss):
            self.incumbent_configs.append(T[0])
            self.incumbent_perfs.append(incumbent_loss)

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
