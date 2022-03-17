# License: MIT

import numpy as np
import time
from openbox.apps.multi_fidelity.mq_base_facade import mqBaseFacade
from openbox.apps.multi_fidelity.utils import sample_configurations

from openbox.utils.config_space import ConfigurationSpace


class mqRandomSearch(mqBaseFacade):
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 n_workers=1,
                 num_iter=10000,
                 random_state=1,
                 method_id='mqRandomSearch',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        max_queue_len = max(100, 3 * n_workers)  # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)
        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)
        self.R = R
        self.n_workers = n_workers
        self.num_iter = num_iter
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
        self.best_config = None
        self.incumbent_configs = []
        self.incumbent_obj = []

    def run(self):
        try:
            for iter in range(1, 1 + self.num_iter):
                self.logger.info('-' * 50)
                self.logger.info("Random Search algorithm: %d/%d iteration starts" % (iter, self.num_iter))
                start_time = time.time()
                self.iterate()
                time_elapsed = (time.time() - start_time) / 60
                self.logger.info("iteration took %.2f min." % time_elapsed)
                self.save_intemediate_statistics()
        except Exception as e:
            print(e)
            self.logger.error(str(e))
            # clear the immediate result.
            # self.remove_immediate_model()

    def iterate(self):
        configs = sample_configurations(self.config_space, self.n_workers)
        extra_info = None
        ret_val, early_stops = self.run_in_parallel(configs, self.R, extra_info)
        val_losses = [item['loss'] for item in ret_val]

        self.incumbent_configs.extend(configs)
        self.incumbent_obj.extend(val_losses)
        self.add_stage_history(self.stage_id, self.global_incumbent)
        self.stage_id += 1
        # self.remove_immediate_model()

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_obj) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_obj)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        targets = [self.incumbent_obj[i] for i in indices[0: num_inc]]
        return configs, targets
