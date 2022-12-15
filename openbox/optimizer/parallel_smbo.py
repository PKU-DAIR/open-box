# License: MIT

import sys
import time
import traceback
from typing import List
from multiprocessing import Lock
import numpy as np

from openbox import logger
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.core.computation.parallel_process import ParallelEvaluation
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.core.ea_advisor import EA_Advisor
from openbox.utils.history import Observation, History
from openbox.optimizer.base import BOBase


def wrapper(param):
    objective_function, config, time_limit_per_trial = param
    trial_state = SUCCESS
    start_time = time.time()
    try:
        args, kwargs = (config,), dict()
        timeout_status, _result = time_limit(objective_function, time_limit_per_trial,
                                             args=args, kwargs=kwargs)
        if timeout_status:
            raise TimeoutException('Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
        else:
            objectives, constraints, extra_info = parse_result(_result)
    except Exception as e:
        if isinstance(e, TimeoutException):
            trial_state = TIMEOUT
        else:
            traceback.print_exc(file=sys.stdout)
            trial_state = FAILED
        objectives = None
        constraints = None
        extra_info = None
    elapsed_time = time.time() - start_time
    return Observation(
        config=config, objectives=objectives, constraints=constraints,
        trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
    )


class pSMBO(BOBase):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            objective_function,
            config_space,
            num_objectives=1,
            num_constraints=0,
            parallel_strategy='async',
            batch_size=4,
            batch_strategy='default',
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
            transfer_learning_history: List[History] = None,
            logging_dir='logs',
            task_id='OpenBox',
            random_state=None,
            advisor_kwargs: dict = None,
            logger_kwargs: dict = None,
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.FAILED_PERF = [np.inf] * num_objectives
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial,
                         transfer_learning_history=transfer_learning_history, logger_kwargs=logger_kwargs)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size

        advisor_kwargs = advisor_kwargs or {}
        _logger_kwargs = {'force_init': False}  # do not init logger in advisor
        if parallel_strategy == 'sync':
            if sample_strategy in ['random', 'bo']:
                self.config_advisor = SyncBatchAdvisor(config_space,
                                                       num_objectives=num_objectives,
                                                       num_constraints=num_constraints,
                                                       batch_size=batch_size,
                                                       batch_strategy=batch_strategy,
                                                       initial_trials=initial_runs,
                                                       initial_configurations=initial_configurations,
                                                       init_strategy=init_strategy,
                                                       transfer_learning_history=transfer_learning_history,
                                                       optimization_strategy=sample_strategy,
                                                       surrogate_type=surrogate_type,
                                                       acq_type=acq_type,
                                                       acq_optimizer_type=acq_optimizer_type,
                                                       ref_point=ref_point,
                                                       task_id=task_id,
                                                       output_dir=logging_dir,
                                                       random_state=random_state,
                                                       logger_kwargs=_logger_kwargs,
                                                       **advisor_kwargs)
            elif sample_strategy == 'ea':
                assert num_objectives == 1 and num_constraints == 0
                self.config_advisor = EA_Advisor(config_space,
                                                 num_objectives=num_objectives,
                                                 num_constraints=num_constraints,
                                                 optimization_strategy=sample_strategy,
                                                 batch_size=batch_size,
                                                 task_id=task_id,
                                                 output_dir=logging_dir,
                                                 random_state=random_state,
                                                 logger_kwargs=_logger_kwargs,
                                                 **advisor_kwargs)
            else:
                raise ValueError('Unknown sample_strategy: %s' % sample_strategy)
        elif parallel_strategy == 'async':
            self.advisor_lock = Lock()
            if sample_strategy in ['random', 'bo']:
                self.config_advisor = AsyncBatchAdvisor(config_space,
                                                        num_objectives=num_objectives,
                                                        num_constraints=num_constraints,
                                                        batch_size=batch_size,
                                                        batch_strategy=batch_strategy,
                                                        initial_trials=initial_runs,
                                                        initial_configurations=initial_configurations,
                                                        init_strategy=init_strategy,
                                                        transfer_learning_history=transfer_learning_history,
                                                        optimization_strategy=sample_strategy,
                                                        surrogate_type=surrogate_type,
                                                        acq_type=acq_type,
                                                        acq_optimizer_type=acq_optimizer_type,
                                                        ref_point=ref_point,
                                                        task_id=task_id,
                                                        output_dir=logging_dir,
                                                        random_state=random_state,
                                                        logger_kwargs=_logger_kwargs,
                                                        **advisor_kwargs)
            elif sample_strategy == 'ea':
                assert num_objectives == 1 and num_constraints == 0
                self.config_advisor = EA_Advisor(config_space,
                                                 num_objectives=num_objectives,
                                                 num_constraints=num_constraints,
                                                 optimization_strategy=sample_strategy,
                                                 batch_size=batch_size,
                                                 task_id=task_id,
                                                 output_dir=logging_dir,
                                                 random_state=random_state,
                                                 logger_kwargs=_logger_kwargs,
                                                 **advisor_kwargs)
            else:
                raise ValueError('Unknown sample_strategy: %s' % sample_strategy)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

    def callback(self, observation: Observation):
        if observation.objectives is None:
            observation.objectives = self.FAILED_PERF.copy()
        # Report the result, and remove the config from the running queue.
        with self.advisor_lock:
            # Parent process: collect the result and increment id.
            self.config_advisor.update_observation(observation)
            logger.info('Update observation %d: %s.' % (self.iteration_id + 1, str(observation)))
            self.iteration_id += 1  # must increment id after updating

    # TODO: Wrong logic. Need to wait before return?
    def async_run(self):
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            while self.iteration_id < self.max_iterations:
                with self.advisor_lock:
                    _config = self.config_advisor.get_suggestion()
                _param = [self.objective_function, _config, self.time_limit_per_trial]
                # Submit a job to worker.
                proc.process_pool.apply_async(wrapper, (_param,), callback=self.callback)
                while len(self.config_advisor.running_configs) >= self.batch_size:
                    # Wait for workers.
                    time.sleep(0.3)

    # Asynchronously evaluate n configs
    def async_iterate(self, n=1) -> List[Observation]:
        iter_id = 0
        res_list = list()
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            while iter_id < n:
                with self.advisor_lock:
                    _config = self.config_advisor.get_suggestion()
                _param = [self.objective_function, _config, self.time_limit_per_trial]
                # Submit a job to worker.
                res_list.append(proc.process_pool.apply_async(wrapper, (_param,), callback=self.callback))
                while len(self.config_advisor.running_configs) >= self.batch_size:
                    # Wait for workers.
                    time.sleep(0.3)
                iter_id += 1
            for res in res_list:
                res.wait()

        iter_observations = self.get_history().observations[-n:]
        return iter_observations  # type: List[Observation]

    def sync_run(self):
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            batch_num = (self.max_iterations + self.batch_size - 1) // self.batch_size
            if self.batch_size > self.config_advisor.init_num:
                batch_num += 1  # fix bug
            batch_id = 0
            while batch_id < batch_num:
                configs = self.config_advisor.get_suggestions()
                logger.info('Running on %d configs in the %d-th batch.' % (len(configs), batch_id))
                params = [(self.objective_function, config, self.time_limit_per_trial) for config in configs]
                # Wait all workers to complete their corresponding jobs.
                observations = proc.parallel_execute(params)
                # Report their results.
                for idx, observation in enumerate(observations):
                    if observation.objectives is None:
                        observation.objectives = self.FAILED_PERF.copy()
                    self.config_advisor.update_observation(observation)
                    logger.info('In the %d-th batch [%d/%d], observation: %s.'
                                     % (batch_id, idx+1, len(configs), observation))
                batch_id += 1

    def run(self) -> History:
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            self.sync_run()
        return self.get_history()
