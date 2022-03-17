# License: MIT

import time
from typing import List
from collections import OrderedDict

from openbox.utils.constants import MAXINT
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.optimizer.base import BOBase
from openbox.core.message_queue.master_messager import MasterMessager
from openbox.core.base import Observation


class mqSMBO(BOBase):
    def __init__(self, objective_function, config_space,
                 parallel_strategy='async',
                 batch_size=4,
                 batch_strategy='default',
                 num_constraints=0,
                 num_objs=1,
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 time_limit_per_trial=180,
                 surrogate_type='auto',
                 acq_type='auto',
                 acq_optimizer_type='auto',
                 initial_runs=3,
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 ref_point=None,
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 task_id='default_task_id',
                 random_state=None,
                 advisor_kwargs: dict = None,
                 ip="",
                 port=13579,
                 authkey=b'abc',):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial,
                         history_bo_data=history_bo_data)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size
        max_queue_len = max(100, 3 * batch_size)
        self.master_messager = MasterMessager(ip, port, authkey, max_queue_len, max_queue_len)

        advisor_kwargs = advisor_kwargs or {}
        if parallel_strategy == 'sync':
            self.config_advisor = SyncBatchAdvisor(config_space,
                                                   num_objs=num_objs,
                                                   num_constraints=num_constraints,
                                                   batch_size=batch_size,
                                                   batch_strategy=batch_strategy,
                                                   initial_trials=initial_runs,
                                                   initial_configurations=initial_configurations,
                                                   init_strategy=init_strategy,
                                                   history_bo_data=history_bo_data,
                                                   optimization_strategy=sample_strategy,
                                                   surrogate_type=surrogate_type,
                                                   acq_type=acq_type,
                                                   acq_optimizer_type=acq_optimizer_type,
                                                   ref_point=ref_point,
                                                   task_id=task_id,
                                                   output_dir=logging_dir,
                                                   random_state=random_state,
                                                   **advisor_kwargs)
        elif parallel_strategy == 'async':
            self.config_advisor = AsyncBatchAdvisor(config_space,
                                                    num_objs=num_objs,
                                                    num_constraints=num_constraints,
                                                    batch_size=batch_size,
                                                    batch_strategy=batch_strategy,
                                                    initial_trials=initial_runs,
                                                    initial_configurations=initial_configurations,
                                                    init_strategy=init_strategy,
                                                    history_bo_data=history_bo_data,
                                                    optimization_strategy=sample_strategy,
                                                    surrogate_type=surrogate_type,
                                                    acq_type=acq_type,
                                                    acq_optimizer_type=acq_optimizer_type,
                                                    ref_point=ref_point,
                                                    task_id=task_id,
                                                    output_dir=logging_dir,
                                                    random_state=random_state,
                                                    **advisor_kwargs)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

    def async_run(self):
        config_num = 0
        result_num = 0
        while result_num < self.max_iterations:
            # Add jobs to masterQueue.
            while len(self.config_advisor.running_configs) < self.batch_size and config_num < self.max_iterations:
                config_num += 1
                config = self.config_advisor.get_suggestion()
                msg = [config, self.time_limit_per_trial]
                self.logger.info("Master: Add config %d." % config_num)
                self.master_messager.send_message(msg)

            # Get results from workerQueue.
            while True:
                observation = self.master_messager.receive_message()
                if observation is None:
                    # Wait for workers.
                    # self.logger.info("Master: wait for worker results. sleep 1s.")
                    time.sleep(1)
                    break
                # Report result.
                result_num += 1
                if observation.objs is None:
                    observation = Observation(
                        config=observation.config, objs=self.FAILED_PERF, constraints=observation.constraints,
                        trial_state=observation.trial_state, elapsed_time=observation.elapsed_time,
                    )
                self.config_advisor.update_observation(observation)
                self.logger.info('Master: Get %d observation: %s' % (result_num, str(observation)))

    def sync_run(self):
        batch_num = (self.max_iterations + self.batch_size - 1) // self.batch_size
        if self.batch_size > self.config_advisor.init_num:
            batch_num += 1  # fix bug
        batch_id = 0
        while batch_id < batch_num:
            configs = self.config_advisor.get_suggestions()
            # Add batch configs to masterQueue.
            for config in configs:
                msg = [config, self.time_limit_per_trial]
                self.master_messager.send_message(msg)
            self.logger.info('Master: %d-th batch. %d configs sent.' % (batch_id, len(configs)))
            # Get batch results from workerQueue.
            result_num = 0
            result_needed = len(configs)
            while True:
                observation = self.master_messager.receive_message()
                if observation is None:
                    # Wait for workers.
                    # self.logger.info("Master: wait for worker results. sleep 1s.")
                    time.sleep(1)
                    continue
                # Report result.
                result_num += 1
                if observation.objs is None:
                    observation = Observation(
                        config=observation.config, objs=self.FAILED_PERF, constraints=observation.constraints,
                        trial_state=observation.trial_state, elapsed_time=observation.elapsed_time,
                    )
                self.config_advisor.update_observation(observation)
                self.logger.info('Master: In the %d-th batch [%d], observation is: %s'
                                 % (batch_id, result_num, str(observation)))
                if result_num == result_needed:
                    break
            batch_id += 1

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            self.sync_run()
        return self.get_history()
