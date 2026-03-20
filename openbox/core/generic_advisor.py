# License: MIT

import numpy as np

from openbox import logger
from openbox.utils.util_funcs import deprecate_kwarg
from openbox.utils.history import History
from openbox.utils.multi_objective import NondominatedPartitioning
from openbox.utils.early_stop import EarlyStopException
from openbox.core.base import build_acq_func, build_surrogate
from openbox.acq_optimizer import build_acq_optimizer
from openbox.core.base_advisor import BaseAdvisor
from openbox.core.initial_config import InitialConfigProvider


class Advisor(BaseAdvisor):
    """
    Generic Bayesian optimization advisor.

    Parameters
    ----------
    config_space : openbox.space.Space or ConfigSpace.ConfigurationSpace
        Configuration space.
    num_objectives : int, default=1
        Number of objectives in objective function.
    num_constraints : int, default=0
        Number of constraints in objective function.
    initial_trials : int, default=3
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
    transfer_learning_history : List[History], optional
        Historical data for transfer learning.
    warm_start_strategy : str, default='topk'
        How to select configs from transfer learning history:
        - 'no': Do not use warm start even if transfer_learning_history is provided
        - 'best': Select best config from each source
        - 'topk': Select top-k configs from each history
    warm_start_num : int, optional
        Number of configs to extract from transfer learning.
        If None, uses init_num by default.
    rand_prob : float, default=0.1
        Probability to sample random configurations.
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
    ref_point : List[float], optional
        Reference point for calculating hypervolume in multi-objective problem.
        Must be provided if using EHVI based acquisition function.
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
    output_dir : str, default='logs'
        Directory to save log files. If None, no log files will be saved.
    task_id : str, default='OpenBox'
        Task identifier.
    random_state : int
        Random seed for RNG.
    logger_kwargs : dict, optional
        Additional keyword arguments for logger.
    """

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            initial_trials=3,
            initial_configurations=None,
            init_strategy='random_explore_first',
            transfer_learning_history=None,
            warm_start_strategy='topk',
            warm_start_num=None,
            rand_prob=0.1,
            optimization_strategy='bo',
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            ref_point=None,
            early_stop=False,
            early_stop_kwargs=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
            **kwargs,
    ):
        super().__init__(
            config_space=config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            ref_point=ref_point,
            early_stop=early_stop,
            early_stop_kwargs=early_stop_kwargs,
            output_dir=output_dir,
            task_id=task_id,
            random_state=random_state,
            logger_kwargs=logger_kwargs,
            **kwargs,
        )

        # Basic components in Advisor.
        self.rand_prob = rand_prob
        self.optimization_strategy = optimization_strategy

        # Init the basic ingredients in Bayesian optimization.
        self.transfer_learning_history = transfer_learning_history
        self.surrogate_transfer_learning_history = self.setup_space_adapter(self.transfer_learning_history)
        self.surrogate_type = surrogate_type
        self.constraint_surrogate_type = None
        self.acq_type = acq_type
        self.acq_optimizer_type = acq_optimizer_type

        self.init_strategy = init_strategy
        self.initial_configurations = self.create_initial_design(
            init_strategy=init_strategy,
            init_num=initial_trials,
            initial_configurations=initial_configurations,
            transfer_learning_history=transfer_learning_history,
            warm_start_strategy=warm_start_strategy,
            warm_start_num=warm_start_num,
            rng=self.rng,
        )
        self.init_num = len(self.initial_config_provider)


        self.surrogate_model = None
        self.constraint_models = None
        self.acquisition_function = None
        self.acq_optimizer = None
        self.auto_alter_model = False
        self.algo_auto_selection()
        self.check_setup()
        self.setup_bo_basics()

    def create_initial_design(self, init_strategy=None, init_num=None, \
            initial_configurations=None, transfer_learning_history=None, \
            warm_start_strategy='no', warm_start_num=0, rng=None):
        if init_strategy is None:
            init_strategy = self.init_strategy
        if init_num is None:
            init_num = self.init_num
        if rng is None:
            rng = self.rng

        self.initial_config_provider = InitialConfigProvider(
            config_space=self.config_space,
            init_num=init_num,
            init_strategy=init_strategy,
            initial_configurations=initial_configurations,
            transfer_learning_history=transfer_learning_history,
            warm_start_strategy=warm_start_strategy,
            warm_start_num=warm_start_num,
            rng=rng,
        )
        return self.initial_config_provider.config_queue

    def algo_auto_selection(self):
        from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
            CategoricalHyperparameter, OrdinalHyperparameter
        # analyze config space
        cont_types = (UniformFloatHyperparameter, UniformIntegerHyperparameter)
        cat_types = (CategoricalHyperparameter, OrdinalHyperparameter)
        n_cont_hp, n_cat_hp, n_other_hp = 0, 0, 0
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, cont_types):
                n_cont_hp += 1
            elif isinstance(hp, cat_types):
                n_cat_hp += 1
            else:
                n_other_hp += 1
        n_total_hp = n_cont_hp + n_cat_hp + n_other_hp

        info_str = ''

        if self.surrogate_type == 'auto':
            use_tl = self.transfer_learning_history is not None
            self.auto_alter_model = True if not use_tl else False
            if n_total_hp >= 100:
                self.optimization_strategy = 'random'
                self.surrogate_type = 'prf'  # for setup procedure
            elif n_total_hp >= 10:
                self.surrogate_type = 'prf' if not use_tl else 'tlbo_rgpe_prf'
            elif n_cat_hp > n_cont_hp:
                self.surrogate_type = 'prf' if not use_tl else 'tlbo_rgpe_prf'
            else:
                self.surrogate_type = 'gp' if not use_tl else 'tlbo_rgpe_gp'
            info_str += ' surrogate_type: %s.' % self.surrogate_type

        if self.acq_type == 'auto':
            if self.num_objectives == 1:  # single objective
                if self.num_constraints == 0:
                    self.acq_type = 'ei'
                else:   # with constraints
                    self.acq_type = 'eic'
            elif self.num_objectives <= 4:    # multi objective (<=4)
                if self.num_constraints == 0:
                    self.acq_type = 'ehvi'
                else:   # with constraints
                    self.acq_type = 'ehvic'
            else:   # multi objective (>4)
                if self.num_constraints == 0:
                    self.acq_type = 'mesmo'
                else:   # with constraints
                    self.acq_type = 'mesmoc'
                self.surrogate_type = 'gp_rbf'
                info_str = ' surrogate_type: %s.' % self.surrogate_type
            info_str += ' acq_type: %s.' % self.acq_type

        if self.acq_optimizer_type == 'auto':
            if n_cat_hp + n_other_hp == 0:  # todo: support constant hp in scipy optimizer
                self.acq_optimizer_type = 'random_scipy'
            else:
                self.acq_optimizer_type = 'local_random'
            info_str += ' acq_optimizer_type: %s.' % self.acq_optimizer_type

        if info_str != '':
            info_str = '[BO auto selection] ' + info_str
            logger.info(info_str)

    def alter_model(self, history: History):
        if not self.auto_alter_model:
            return

        num_config_evaluated = len(history)

        if num_config_evaluated >= 300:
            if self.surrogate_type == 'gp':
                self.surrogate_type = 'prf'
                logger.info(f'n_observations={num_config_evaluated}, change surrogate model from GP to PRF!')
                if self.acq_optimizer_type == 'random_scipy':
                    self.acq_optimizer_type = 'local_random'
                    logger.info(f'n_observations={num_config_evaluated}, '
                                f'change acq optimizer from random_scipy to local_random!')
                self.setup_bo_basics()

    def check_setup(self):
        """
        Check optimization_strategy, num_objectives, num_constraints, acq_type, surrogate_type.
        Returns
        -------
        None
        """
        assert self.optimization_strategy in ['bo', 'random']
        assert isinstance(self.num_objectives, int) and self.num_objectives >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

        # single objective
        if self.num_objectives == 1:
            if self.num_constraints == 0:
                assert self.acq_type in ['ei', 'eips', 'logei', 'pi', 'lcb', 'lpei', ]
            else:  # with constraints
                assert self.acq_type in ['eic', ]
                if self.constraint_surrogate_type is None:
                    self.constraint_surrogate_type = 'gp'

        # multi-objective
        else:
            if self.num_constraints == 0:
                assert self.acq_type in ['ehvi', 'mesmo', 'usemo', 'parego']
                if self.acq_type == 'ehvi' and self.num_objectives > 4:
                    logger.warning('Computational cost of EHVI with more than 4 objectives might be extremely high!')
                if self.acq_type == 'mesmo' and self.surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                   'since MESMO is used. Surrogate_type should be set to \'gp_rbf\'.')
            else:  # with constraints
                assert self.acq_type in ['ehvic', 'mesmoc', 'mesmoc2']
                if self.constraint_surrogate_type is None:
                    if self.acq_type == 'mesmoc':
                        self.constraint_surrogate_type = 'gp_rbf'
                    else:
                        self.constraint_surrogate_type = 'gp'
                if self.acq_type == 'ehvic' and self.num_objectives > 4:
                    logger.warning('Computational cost of EHVIC with more than 4 objectives might be extremely high!')
                if self.acq_type == 'mesmoc' and self.surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                   'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')
                if self.acq_type == 'mesmoc' and self.constraint_surrogate_type != 'gp_rbf':
                    self.surrogate_type = 'gp_rbf'
                    logger.warning('Constraint surrogate model has changed to Gaussian Process with RBF kernel '
                                   'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')

            # Check reference point is provided for EHVI methods
            if 'ehvi' in self.acq_type and self.ref_point is None:
                raise ValueError('Must provide reference point to use EHVI method!')

        # transfer learning
        if self.transfer_learning_history is not None:
            if not (self.num_objectives == 1 and self.num_constraints == 0):
                raise NotImplementedError('Currently, transfer learning is only supported for single objective '
                                          'optimization without constraints.')
            if self.surrogate_type.startswith('mfgpe'):
                pass
            else:
                surrogate_str = self.surrogate_type.split('_')
                assert len(surrogate_str) == 3 and surrogate_str[0] == 'tlbo'
                assert surrogate_str[1] in ['rgpe', 'sgpr', 'topov3', 'mfgpe']

        # early stop
        if self.early_stop:
            self.early_stop_algorithm.check_setup(advisor=self)

    def setup_bo_basics(self):
        """
        Prepare the basic BO components.
        Returns
        -------
        An optimizer object.
        """
        if self.num_objectives == 1:
            self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                                   config_space=self.surrogate_space,
                                                   rng=self.rng,
                                                   transfer_learning_history=self.surrogate_transfer_learning_history)
        elif self.acq_type == 'parego':
            func_str = 'parego_' + self.surrogate_type
            self.surrogate_model = build_surrogate(func_str=func_str,
                                                   config_space=self.surrogate_space,
                                                   rng=self.rng,
                                                   transfer_learning_history=self.surrogate_transfer_learning_history)
        else:  # multi-objectives
            self.surrogate_model = [build_surrogate(func_str=self.surrogate_type,
                                                    config_space=self.surrogate_space,
                                                    rng=self.rng,
                                                    transfer_learning_history=self.surrogate_transfer_learning_history)
                                    for _ in range(self.num_objectives)]

        if self.num_constraints > 0:
            self.constraint_models = [build_surrogate(func_str=self.constraint_surrogate_type,
                                                      config_space=self.config_space,
                                                      rng=self.rng) for _ in range(self.num_constraints)]

        if self.acq_type in ['mesmo', 'mesmoc', 'mesmoc2', 'usemo']:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       config_space=self.surrogate_space)
        else:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       ref_point=self.ref_point)
        if self.acq_type == 'usemo':
            self.acq_optimizer_type = 'usemo_optimizer'
        self.acq_optimizer = build_acq_optimizer(
            func_str=self.acq_optimizer_type, config_space=self.sample_space, rng=self.rng)

    def early_stop_ei(self, history, challengers):
        if not self.early_stop:
            return

        if self.early_stop_algorithm.already_early_stopped(history):
            raise EarlyStopException("Early stop triggered!")

        max_acq_value = np.max(self.acquisition_function(challengers)).item()
        if self.early_stop_algorithm.decide_early_stop_after_suggest(
                history=history, max_acq_value=max_acq_value):
            self.early_stop_algorithm.set_already_early_stopped(history)
            raise EarlyStopException("Early stop triggered!")

    def update_compression(self, history: History = None) -> bool:
        if history is None:
            history = self.history
        if not self.space_adapter.update(history):
            return False
        self.sample_space = self.space_adapter.sample_space
        self.surrogate_space = self.space_adapter.surrogate_space
        self.surrogate_transfer_learning_history = self.space_adapter.setup(self.transfer_learning_history)
        self.setup_bo_basics()
        return True

    def _get_bo_candidates(self, history: History):
        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()

        if num_config_successful < max(self.init_num, 1):
            logger.warning('No enough successful initial trials! Sample random configuration.')
            return self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)

        X = self.space_adapter.get_surrogate_array(history)
        Y = history.get_objectives(transform='infeasible')
        cY = history.get_constraints(transform='bilog')

        # train surrogate model
        if self.num_objectives == 1:
            self.surrogate_model.train(X, Y[:, 0])
        elif self.acq_type == 'parego':
            self.surrogate_model.train(X, Y)
        else:  # multi-objectives
            for i in range(self.num_objectives):
                self.surrogate_model[i].train(X, Y[:, i])

        # train constraint model
        for i in range(self.num_constraints):
            self.constraint_models[i].train(X, cY[:, i])

        # update acquisition function
        if self.num_objectives == 1:
            incumbent_value = history.get_incumbent_value()
            self.acquisition_function.update(model=self.surrogate_model,
                                             constraint_models=self.constraint_models,
                                             eta=incumbent_value,
                                             num_data=num_config_evaluated)
        else:  # multi-objectives
            mo_incumbent_values = history.get_mo_incumbent_values()
            if self.acq_type == 'parego':
                scalarized_obj = self.surrogate_model.get_scalarized_obj()
                incumbent_value = scalarized_obj(np.atleast_2d(mo_incumbent_values))
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value,
                                                 num_data=num_config_evaluated)
            elif self.acq_type.startswith('ehvi'):
                partitioning = NondominatedPartitioning(self.num_objectives, Y)
                cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 cell_lower_bounds=cell_bounds[0],
                                                 cell_upper_bounds=cell_bounds[1])
            else:
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 constraint_perfs=cY,  # for MESMOC
                                                 eta=mo_incumbent_values,
                                                 num_data=num_config_evaluated,
                                                 X=X, Y=Y)

        challengers = self.acq_optimizer.maximize(
            acquisition_function=self.acquisition_function,
            history=self.space_adapter.history_for_acq(history),
            num_points=5000,
        )
        return [self.space_adapter.config_to_original(conf) for conf in challengers]

    def get_suggestion(self, history: History = None):
        if history is None:
            history = self.history

        self.early_stop_perf(history)
        self.alter_model(history)

        num_config_evaluated = len(history)
        if num_config_evaluated < self.init_num:
            return self.initial_config_provider.get_config(num_config_evaluated)
        if self.optimization_strategy == 'random':
            return self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]

        if self.rng.random() < self.rand_prob:
            logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
            return self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]

        if self.optimization_strategy != 'bo':
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

        candidates = self._get_bo_candidates(history)
        self.early_stop_ei(history, challengers=candidates)
        for config in candidates:
            if config not in history.configurations:
                return config
        logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                       'Sample random config.' % (len(candidates),))
        return self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]


    def get_suggestions(self, batch_size=None, history: History = None):
        if batch_size is None:
            batch_size = 1
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []
        if history is None:
            history = self.history

        self.early_stop_perf(history)
        self.alter_model(history)

        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()
        if (
            num_config_evaluated < self.init_num
            or self.optimization_strategy == 'random'
            or num_config_successful < max(self.init_num, 1)
        ):
            return self.sample_random_configs(
                self.config_space,
                num_configs=batch_size,
                excluded_configs=history.configurations,
            )

        if self.optimization_strategy != 'bo':
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

        candidates = self._get_bo_candidates(history)
        self.early_stop_ei(history, challengers=candidates)

        results = []
        for config in candidates:
            if config in history.configurations or config in results:
                continue
            results.append(config)
            if len(results) >= batch_size:
                return results

        if len(results) < batch_size:
            excluded = set(history.configurations)
            excluded.update(results)
            results.extend(self.sample_random_configs(
                self.config_space,
                num_configs=batch_size - len(results),
                excluded_configs=excluded,
            ))
        return results
