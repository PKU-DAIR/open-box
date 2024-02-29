# License: MIT

import time
from typing import List
from tqdm import tqdm
import numpy as np
from openbox import logger
from openbox.optimizer.base import BOBase
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.early_stop import EarlyStopException
from openbox.visualization import build_visualizer


class SMBO(BOBase):
    """
    Generic Optimizer

    Parameters
    ----------
    objective_function : callable
        Objective function to optimize.
    config_space : openbox.space.Space or ConfigSpace.ConfigurationSpace
        Configuration space.
    num_objectives : int, default=1
        Number of objectives in objective function.
    num_constraints : int, default=0
        Number of constraints in objective function.
    max_runs : int
        Number of optimization iterations.
    max_runtime : int or float, optional
        Time budget for the whole optimization process. None means no limit.
    max_runtime_per_trial : int or float, optional
        Time budget for a single evaluation trial. None means no limit.
    advisor_type : str
        Type of advisor to produce configuration suggestion.
        - 'default' (default): Bayesian Optimization
        - 'tpe': Tree-structured Parzen Estimator
        - 'ea': Evolutionary Algorithms
        - 'random': Random Search
        - 'mcadvisor': Bayesian Optimization with Monte Carlo Sampling
    surrogate_type : str, default='auto'
        Type of surrogate model in Bayesian optimization.
        - 'gp' (default): Gaussian Process. Better performance for mathematical problems.
        - 'prf': Probability Random Forest. Better performance for hyper-parameter optimization (HPO).
        - 'lightgbm': LightGBM.
    acq_type : str, default='auto'
        Type of acquisition function in Bayesian optimization.
        For single objective problem:
        - 'ei' (default): Expected Improvement
        - 'eips': Expected Improvement per Second
        - 'logei': Logarithm Expected Improvement
        - 'pi': Probability of Improvement
        - 'lcb': Lower Confidence Bound
        For single objective problem with constraints:
        - 'eic' (default): Expected Constrained Improvement
        For multi-objective problem:
        - 'ehvi (default)': Expected Hypervolume Improvement
        - 'mesmo': Multi-Objective Max-value Entropy Search
        - 'usemo': Multi-Objective Uncertainty-Aware Search
        - 'parego': ParEGO
        For multi-objective problem with constraints:
        - 'ehvic' (default): Expected Hypervolume Improvement with Constraints
        - 'mesmoc': Multi-Objective Max-value Entropy Search with Constraints
    acq_optimizer_type : str, default='auto'
        Type of optimizer to maximize acquisition function.
        - 'local_random' (default): Interleaved Local and Random Search
        - 'random_scipy': L-BFGS-B (Scipy) optimizer with random starting points
        - 'scipy_global': Differential Evolution
        - 'cma_es': Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    initial_runs : int, default=3
        Number of initial iterations of optimization.
    init_strategy : str, default='random_explore_first'
        Strategy to generate configurations for initial iterations.
        - 'random_explore_first' (default): Random sampled configs with maximized internal minimum distance
        - 'random': Random sampling
        - 'default': Default configuration + random sampling
        - 'sobol': Sobol sequence sampling
        - 'latin_hypercube': Latin hypercube sampling
    initial_configurations : List[Configuration], optional
        If provided, the initial configurations will be evaluated in initial iterations of optimization.
    ref_point : List[float], optional
        Reference point for calculating hypervolume in multi-objective problem.
        Must be provided if using EHVI based acquisition function.
    transfer_learning_history : List[History], optional
        Historical data for transfer learning.
    early_stop: bool, default=False
        Whether to enable early stop.
    early_stop_kwargs : dict, optional
        Options for early stop algorithm:
        - min_iter : int
            Minimum number of iterations before early stop is considered.
        - min_improvement_percentage : float
            The minimum improvement percentage. If the Expected Improvement (EI) is less than
            `min_improvement_percentage * (default_obj_value - best_obj_value)`, early stop is triggered.
            If `improvement_threshold` is 0, this criterion is disabled.
        - max_no_improvement_rounds : int
            The maximum tolerable rounds with no improvement before early stop.
            If `max_no_improvement_rounds` is 0, this criterion is disabled.
    logging_dir : str, default='logs'
        Directory to save log files. If None, no log files will be saved.
    task_id : str, default='OpenBox'
        Task identifier.
    visualization : ['none', 'basic', 'advanced'], default='none'
        HTML visualization option.
        - 'none': Run the task without visualization. No additional files are generated.
                  Better for running massive experiments.
        - 'basic': Run the task with basic visualization, including basic charts for objectives and constraints.
        - 'advanced': Enable visualization with advanced functions,
                      including surrogate fitting analysis and hyperparameter importance analysis.
    auto_open_html : bool, default=False
        Whether to automatically open the HTML file for visualization. Only works when `visualization` is not 'none'.
    random_state : int
        Random seed for RNG.
    logger_kwargs : dict, optional
        Additional keyword arguments for logger.
    advisor_kwargs : dict, optional
        Additional keyword arguments for advisor.
    """
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    @deprecate_kwarg('time_limit_per_trial', 'max_runtime_per_trial', 'a future version')
    @deprecate_kwarg('runtime_limit', 'max_runtime', 'a future version')
    def __init__(
            self,
            objective_function: callable,
            config_space,
            num_objectives=1,
            num_constraints=0,
            sample_strategy: str = 'bo',
            max_runs=100,
            max_runtime=None,
            max_runtime_per_trial=None,
            advisor_type='default',
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            initial_runs=3,
            init_strategy='random_explore_first',
            initial_configurations=None,
            ref_point=None,
            transfer_learning_history: List[History] = None,
            early_stop=False,
            early_stop_kwargs=None,
            logging_dir='logs',
            task_id='OpenBox',
            visualization='none',
            auto_open_html=False,
            random_state=None,
            logger_kwargs: dict = None,
            advisor_kwargs: dict = None,
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.FAILED_PERF = [np.inf] * num_objectives
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         max_runtime=max_runtime, max_runtime_per_trial=max_runtime_per_trial,
                         sample_strategy=sample_strategy, transfer_learning_history=transfer_learning_history,
                         logger_kwargs=logger_kwargs)

        self.advisor_type = advisor_type
        advisor_kwargs = advisor_kwargs or {}
        _logger_kwargs = {'force_init': False}  # do not init logger in advisor
        if advisor_type == 'default':
            from openbox.core.generic_advisor import Advisor
            self.config_advisor = Advisor(config_space,
                                          num_objectives=num_objectives,
                                          num_constraints=num_constraints,
                                          initial_trials=initial_runs,
                                          init_strategy=init_strategy,
                                          initial_configurations=initial_configurations,
                                          optimization_strategy=sample_strategy,
                                          surrogate_type=surrogate_type,
                                          acq_type=acq_type,
                                          acq_optimizer_type=acq_optimizer_type,
                                          ref_point=ref_point,
                                          transfer_learning_history=transfer_learning_history,
                                          early_stop=early_stop,
                                          early_stop_kwargs=early_stop_kwargs,
                                          task_id=task_id,
                                          output_dir=logging_dir,
                                          random_state=random_state,
                                          logger_kwargs=_logger_kwargs,
                                          **advisor_kwargs)
        elif advisor_type == 'mcadvisor':
            from openbox.core.mc_advisor import MCAdvisor
            self.config_advisor = MCAdvisor(config_space,
                                            num_objectives=num_objectives,
                                            num_constraints=num_constraints,
                                            initial_trials=initial_runs,
                                            init_strategy=init_strategy,
                                            initial_configurations=initial_configurations,
                                            optimization_strategy=sample_strategy,
                                            surrogate_type=surrogate_type,
                                            acq_type=acq_type,
                                            acq_optimizer_type=acq_optimizer_type,
                                            ref_point=ref_point,
                                            transfer_learning_history=transfer_learning_history,
                                            task_id=task_id,
                                            output_dir=logging_dir,
                                            random_state=random_state,
                                            logger_kwargs=_logger_kwargs,
                                            **advisor_kwargs)
        elif advisor_type == 'tpe':
            from openbox.core.tpe_advisor import TPE_Advisor
            assert num_objectives == 1 and num_constraints == 0
            self.config_advisor = TPE_Advisor(config_space, task_id=task_id, random_state=random_state,
                                              logger_kwargs=_logger_kwargs, **advisor_kwargs)
        elif advisor_type == 'ea':
            from openbox.core.ea_advisor import EA_Advisor
            assert num_objectives == 1 and num_constraints == 0
            self.config_advisor = EA_Advisor(config_space,
                                             num_objectives=num_objectives,
                                             num_constraints=num_constraints,
                                             optimization_strategy=sample_strategy,
                                             batch_size=1,
                                             task_id=task_id,
                                             output_dir=logging_dir,
                                             random_state=random_state,
                                             logger_kwargs=_logger_kwargs,
                                             **advisor_kwargs)
        elif advisor_type == 'random':
            from openbox.core.random_advisor import RandomAdvisor
            self.config_advisor = RandomAdvisor(config_space,
                                                num_objectives=num_objectives,
                                                num_constraints=num_constraints,
                                                initial_trials=initial_runs,
                                                init_strategy=init_strategy,
                                                initial_configurations=initial_configurations,
                                                surrogate_type=surrogate_type,
                                                acq_type=acq_type,
                                                acq_optimizer_type=acq_optimizer_type,
                                                ref_point=ref_point,
                                                transfer_learning_history=transfer_learning_history,
                                                task_id=task_id,
                                                output_dir=logging_dir,
                                                random_state=random_state,
                                                logger_kwargs=_logger_kwargs,
                                                **advisor_kwargs)
        else:
            raise ValueError('Invalid advisor type!')

        self.visualizer = build_visualizer(
            option=visualization, history=self.get_history(),
            logging_dir=self.output_dir, optimizer=self, advisor=None, auto_open_html=auto_open_html,
        )
        self.visualizer.setup()

    def run(self) -> History:
        for idx in tqdm(range(self.iteration_id, self.max_runs)):
            if self.time_left <= 0:
                logger.info(f'max runtime ({self.max_runtime}s) exceeded, stop optimization.')
                break
            start_time = time.time()
            try:
                self.iterate(time_left=self.time_left)
            except EarlyStopException:
                logger.info(f'Early stop triggered at iter {idx}!')
                break
            runtime = time.time() - start_time
            self.time_left -= runtime
        return self.get_history()

    def iterate(self, time_left=None) -> Observation:
        # get configuration suggestion from advisor
        config = self.config_advisor.get_suggestion()

        if config in self.config_advisor.history.configurations:
            logger.warning('Evaluating duplicated configuration: %s' % config)

        if time_left is None:
            timeout = self.max_runtime_per_trial
        elif self.max_runtime_per_trial is None:
            timeout = time_left
        else:
            timeout = min(time_left, self.max_runtime_per_trial)
        if np.isinf(timeout):
            timeout = None

        # evaluate configuration on objective_function
        obj_args, obj_kwargs = (config,), dict()
        result = run_obj_func(self.objective_function, obj_args, obj_kwargs, timeout)

        # parse result
        ret, timeout_status, traceback_msg, elapsed_time = (
            result['result'], result['timeout'], result['traceback'], result['elapsed_time'])
        if timeout_status:
            trial_state = TIMEOUT
        elif traceback_msg is not None:
            trial_state = FAILED
            logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
        else:
            trial_state = SUCCESS
        if trial_state == SUCCESS:
            objectives, constraints, extra_info = parse_result(ret)
        else:
            objectives, constraints, extra_info = self.FAILED_PERF.copy(), None, None

        # update observation to advisor
        observation = Observation(
            config=config, objectives=objectives, constraints=constraints,
            trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
        )
        self.config_advisor.update_observation(observation)

        self.iteration_id += 1
        # Logging
        if self.num_constraints > 0:
            logger.info('Iter %d, objectives: %s. constraints: %s.' % (self.iteration_id, objectives, constraints))
        else:
            logger.info('Iter %d, objectives: %s.' % (self.iteration_id, objectives))

        self.visualizer.update()
        return observation
