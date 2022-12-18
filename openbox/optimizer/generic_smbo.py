# License: MIT

import time
import math
from typing import List
from tqdm import tqdm
import numpy as np
from openbox import logger
from openbox.optimizer.base import BOBase
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from openbox.utils.history import Observation, History
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
    runtime_limit : int or float, optional
        Time budget for the whole optimization process. None means no limit.
    time_limit_per_trial : int or float
        Time budget for a single evaluation trial.
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
    def __init__(
            self,
            objective_function: callable,
            config_space,
            num_objectives=1,
            num_constraints=0,
            sample_strategy: str = 'bo',
            max_runs=200,
            runtime_limit=None,
            time_limit_per_trial=180,
            advisor_type='default',
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
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, transfer_learning_history=transfer_learning_history,
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

        self.visualizer = build_visualizer(visualization, self, auto_open_html=auto_open_html)
        self.visualizer.setup()

    def run(self) -> History:
        for _ in tqdm(range(self.iteration_id, self.max_iterations)):
            if self.budget_left < 0:
                logger.info('Time %f elapsed!' % self.runtime_limit)
                break
            start_time = time.time()
            self.iterate(budget_left=self.budget_left)
            runtime = time.time() - start_time
            self.budget_left -= runtime
        return self.get_history()

    def iterate(self, budget_left=None) -> Observation:
        # get configuration suggestion from advisor
        config = self.config_advisor.get_suggestion()

        trial_state = SUCCESS
        _budget_left = int(1e10) if budget_left is None else budget_left
        _time_limit_per_trial = math.ceil(min(self.time_limit_per_trial, _budget_left))

        if config in self.config_advisor.history.configurations:
            logger.warning('Evaluating duplicated configuration: %s' % config)

        start_time = time.time()
        try:
            # evaluate configuration on objective_function within time_limit_per_trial
            args, kwargs = (config,), dict()
            timeout_status, _result = time_limit(self.objective_function,
                                                 _time_limit_per_trial,
                                                 args=args, kwargs=kwargs)
            if timeout_status:
                raise TimeoutException(
                    'Timeout: time limit for this evaluation is %.1fs' % _time_limit_per_trial)
            else:
                # parse result
                objectives, constraints, extra_info = parse_result(_result)
        except Exception as e:
            # parse result of failed trial
            if isinstance(e, TimeoutException):
                logger.warning(str(e))
                trial_state = TIMEOUT
            else:  # todo: log exception if objective function raises error
                logger.warning(f'Exception when calling objective function: {e}\nconfig: {config}')
                trial_state = FAILED
            objectives = self.FAILED_PERF
            constraints = None
            extra_info = None

        elapsed_time = time.time() - start_time
        # update observation to advisor
        observation = Observation(
            config=config, objectives=objectives, constraints=constraints,
            trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
        )
        if _time_limit_per_trial != self.time_limit_per_trial and trial_state == TIMEOUT:
            # Timeout in the last iteration.
            pass
        else:
            self.config_advisor.update_observation(observation)

        self.iteration_id += 1
        # Logging
        if self.num_constraints > 0:
            logger.info('Iter %d, objectives: %s. constraints: %s.'
                             % (self.iteration_id, objectives, constraints))
        else:
            logger.info('Iter %d, objectives: %s.' % (self.iteration_id, objectives))

        self.visualizer.update()
        return observation
