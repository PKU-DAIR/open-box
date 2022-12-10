# License: MIT

import time
import math
from typing import List
from collections import OrderedDict
from tqdm import tqdm
from openbox import logger
from openbox.optimizer.base import BOBase
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from openbox.utils.history_container import Observation
from openbox.visualization import build_visualizer


class SMBO(BOBase):
    """
    Parameters
    ----------
    objective_function : callable
        Objective function to optimize.
    config_space : openbox.space.Space
        Configuration space.
    num_objectives : int
        Number of objectives in objective function.
    num_constraints : int
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
    surrogate_type : str
        Type of surrogate model in Bayesian optimization.
        - 'gp' (default): Gaussian Process. Better performance for mathematical problems.
        - 'prf': Probability Random Forest. Better performance for hyper-parameter optimization (HPO).
        - 'lightgbm': LightGBM.
    acq_type : str
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
    acq_optimizer_type : str
        Type of optimizer to maximize acquisition function.
        - 'local_random' (default): Interleaved Local and Random Search
        - 'random_scipy': L-BFGS-B (Scipy) optimizer with random starting points
        - 'scipy_global': Differential Evolution
        - 'cma_es': Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    initial_runs : int
        Number of initial iterations of optimization.
    init_strategy : str
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
    history_bo_data : List[OrderedDict], optional
        Historical data for transfer learning.
    logging_dir : str
        Directory to save log files.
    task_id : str
        Task identifier.
    visualization : ['none', 'basic', 'advanced']
        HTML visualization option
    random_state : int
        Random seed for RNG.
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
            history_bo_data: List[OrderedDict] = None,
            logging_dir='logs',
            task_id='OpenBox',
            visualization='none',
            random_state=None,
            logger_kwargs: dict = None,
            advisor_kwargs: dict = None,
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.FAILED_PERF = [MAXINT] * num_objectives
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, history_bo_data=history_bo_data,
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
                                          history_bo_data=history_bo_data,
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
                                            history_bo_data=history_bo_data,
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
                                                history_bo_data=history_bo_data,
                                                task_id=task_id,
                                                output_dir=logging_dir,
                                                random_state=random_state,
                                                logger_kwargs=_logger_kwargs,
                                                **advisor_kwargs)
        else:
            raise ValueError('Invalid advisor type!')

        self.visualizer = build_visualizer(visualization, self)
        self.visualizer.setup()

    def run(self):
        for _ in tqdm(range(self.iteration_id, self.max_iterations)):
            if self.budget_left < 0:
                logger.info('Time %f elapsed!' % self.runtime_limit)
                break
            start_time = time.time()
            self.iterate(budget_left=self.budget_left)
            runtime = time.time() - start_time
            self.budget_left -= runtime
        return self.get_history()

    def iterate(self, budget_left=None):
        # get configuration suggestion from advisor
        config = self.config_advisor.get_suggestion()

        trial_state = SUCCESS
        _budget_left = int(1e10) if budget_left is None else budget_left
        _time_limit_per_trial = math.ceil(min(self.time_limit_per_trial, _budget_left))

        # only evaluate non duplicate configuration
        if config not in self.config_advisor.history_container.configurations:
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
                else:
                    logger.warning('Exception when calling objective function: %s' % str(e))
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
        else:
            logger.info('This configuration has been evaluated! Skip it: %s' % config)
            history = self.get_history()
            config_idx = history.configurations.index(config)
            trial_state = history.trial_states[config_idx]
            objectives = history.perfs[config_idx]
            constraints = history.constraint_perfs[config_idx] if self.num_constraints > 0 else None
            if self.num_objectives == 1:
                objectives = (objectives,)

        self.iteration_id += 1
        # Logging
        if self.num_constraints > 0:
            logger.info('Iter %d, objectives: %s. constraints: %s.'
                             % (self.iteration_id, objectives, constraints))
        else:
            logger.info('Iter %d, objectives: %s.' % (self.iteration_id, objectives))

        self.visualizer.update()
        return config, trial_state, constraints, objectives
