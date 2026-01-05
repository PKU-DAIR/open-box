# License: MIT

import time
import inspect
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
    scheduler_type : str, default='full'
        Type of fidelity scheduler for multi-fidelity optimization.
        - 'full' (default): Full fidelity scheduler (resource_ratio=1.0). Behaves like standard single-fidelity BO.
        - 'bohb': BOHB-style successive halving scheduler
        - 'flatten': Flattened BOHB scheduler with expanded full-fidelity brackets
        - 'mfes': MFES-style multi-fidelity scheduler  
        - 'mfes_flatten': Flattened MFES scheduler
        - 'fixed': Fixed fidelity levels (requires scheduler_kwargs)
    scheduler_kwargs : dict, optional
        Additional keyword arguments for scheduler initialization.
        For BOHB/MFES schedulers:
        - R : int, default=9
            Maximum resource allocation
        - eta : int, default=3
            Reduction factor for successive halving
        - num_nodes : int, default=1
            Number of parallel nodes (for distributed optimization)
        For fixed scheduler:
        - n_resources : List[int]
            Number of configurations at each stage
        - r_resources : List[int]
            Resource allocations at each stage
        - fidelity_levels : List[float]
            Available fidelity levels
    
    Notes
    -----
    Multi-Fidelity Optimization:
        When using scheduler_type other than 'full', the objective function should accept
        a `resource_ratio` keyword argument (float, 0.0 to 1.0) to control the evaluation fidelity.
        For example, in hyperparameter optimization:
        - resource_ratio=0.1: Train on 10% of data
        - resource_ratio=1.0: Train on full dataset
        
        The optimizer function signature should be:
        def objective_function(config, resource_ratio=1.0):
            # Use resource_ratio to control fidelity
            ...
            return result
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
            scheduler_type: str = 'full',
            scheduler_kwargs: dict = None,
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
        
        from openbox.optimizer.scheduler import build_scheduler, check_scheduler
        scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_type = scheduler_type
        self.scheduler = build_scheduler(scheduler_type, **scheduler_kwargs)
        logger.info(f'Using scheduler: {scheduler_type} with fidelity levels: '
                   f'{self.scheduler.get_fidelity_levels()}')
        # Check if objective function supports resource_ratio for multi-fidelity optimization
        self._supports_resource_ratio = check_scheduler(objective_function, scheduler_type)

        from openbox.core import build_advisor
        self.config_advisor = build_advisor(
            advisor_type=advisor_type,
            config_space=config_space,
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
            logger_kwargs={'force_init': False},  # do not init logger in advisor
            scheduler_type=scheduler_type,
            **advisor_kwargs
        )

        self.visualizer = build_visualizer(
            option=visualization, history=self.get_history(),
            logging_dir=self.output_dir, optimizer=self, advisor=None, auto_open_html=auto_open_html,
        )
        self.visualizer.setup()
        self._fidelity_support = self._check_fidelity_support()

    def _check_fidelity_support(self) -> bool:
        try:
            signature = inspect.signature(self.config_advisor.update_observation)
        except (TypeError, ValueError):
            return False
        return 'resource_ratio' in signature.parameters

    def _update_advi_obs(self, observation: Observation, resource_ratio: float):
        if self._fidelity_support:
            self.config_advisor.update_observation(observation, resource_ratio=resource_ratio)
        else:
            self.config_advisor.update_observation(observation)

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

    def _evaluate_single_config(self, config, resource_ratio=1.0, time_left=None) -> Observation:
        """
        Evaluate a single configuration with specified resource ratio.
        
        Args:
            config: Configuration to evaluate
            resource_ratio: Resource ratio for multi-fidelity (0.0 to 1.0)
            time_left: Remaining time budget
            
        Returns:
            Observation object containing evaluation results
        """
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
        # pass resource_ratio to objective function for multi-fidelity support
        if self._supports_resource_ratio:
            obj_args, obj_kwargs = (config,), dict(resource_ratio=resource_ratio)
        else:
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
        
        return observation

    def iterate(self, time_left=None) -> Observation:
        self.iteration_id += 1
        
        # Initial runs: always use full fidelity (resource_ratio=1.0)
        if self.iteration_id <= self.config_advisor.init_num:
            configs = self.config_advisor.get_suggestions(batch_size=1)
            config = configs[0]
            observation = self._evaluate_single_config(config, resource_ratio=1.0, time_left=time_left)
            self._update_advi_obs(observation, resource_ratio=1.0)
            
            if self.num_constraints > 0:
                logger.info('Iter %d (init), objectives: %s, constraints: %s, resource_ratio: 1.0' % 
                          (self.iteration_id, observation.objectives, observation.constraints))
            else:
                logger.info('Iter %d (init), objectives: %s, resource_ratio: 1.0' % 
                          (self.iteration_id, observation.objectives))
            
            self.visualizer.update()
            return observation
        
        # After initialization: use scheduler for multi-fidelity optimization
        iter_full_eval_observations = []
        candidates = []
        
        # Get bracket index based on current iteration
        s = self.scheduler.get_bracket_index(self.iteration_id - self.config_advisor.init_num - 1)
        
        # Execute successive halving within the bracket
        # For 'full' scheduler: s=0, only 1 stage, 1 config, ratio=1.0
        for stage in range(s + 1):
            n_configs, n_resource = self.scheduler.get_stage_params(s=s, stage=stage)
            resource_ratio = self.scheduler.calculate_resource_ratio(n_resource=n_resource)
            
            if self.scheduler_type != 'full':
                logger.info(f'Bracket {s} Stage {stage}: n_configs={n_configs}, '
                           f'resource={n_resource}, ratio={resource_ratio:.3f}')
            
            # First stage: sample new configurations
            if stage == 0:
                candidates = self.config_advisor.get_suggestions(batch_size=n_configs)
                if self.scheduler_type != 'full' and len(candidates) > 1:
                    logger.info(f'Generated {len(candidates)} initial candidates for stage {stage}')
            
            # Evaluate all candidates at current fidelity
            observations = []
            perfs = []
            for config in candidates:
                obs = self._evaluate_single_config(config, resource_ratio, time_left)
                observations.append(obs)
                perfs.append(obs.objectives[0])  # Use first objective for elimination
                
                # Update advisor if scheduler says so
                if self.scheduler.should_update_history(resource_ratio):
                    self._update_advi_obs(obs, resource_ratio=resource_ratio)
                    if self.num_constraints > 0:
                        logger.info('Iter %d, objectives: %s, constraints: %s, '
                                    'resource_ratio: %.3f' % 
                                    (self.iteration_id, obs.objectives, obs.constraints, resource_ratio))
                    else:
                        logger.info('Iter %d, objectives: %s, resource_ratio: %.3f' % 
                                    (self.iteration_id, obs.objectives, resource_ratio))
                    
            # Eliminate poor performing candidates for next stage
            if stage < s:
                candidates, perfs = self.scheduler.eliminate_candidates(
                    candidates, perfs, s=s, stage=stage
                )
                logger.info(f'After elimination: {len(candidates)} candidates remain')
            else:
                # Last stage: these are full-fidelity evaluations
                iter_full_eval_observations.extend(observations)
        
        self.visualizer.update()
        return iter_full_eval_observations[-1] if iter_full_eval_observations else None
