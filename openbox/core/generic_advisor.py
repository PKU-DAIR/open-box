# License: MIT

import os
import abc
import numpy as np
from datetime import datetime

from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.samplers import SobolSampler, LatinHypercubeSampler, HaltonSampler
from openbox.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
from openbox.core.base import build_acq_func, build_optimizer, build_surrogate


class Advisor(object, metaclass=abc.ABCMeta):
    """
    Basic Advisor Class, which adopts a policy to sample a configuration.
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
            rand_prob=0.1,
            optimization_strategy='bo',
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            ref_point=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
            **kwargs,
    ):

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.init_strategy = init_strategy
        self.output_dir = output_dir
        self.task_id = task_id
        self.rng = check_random_state(random_state)

        _logger_kwargs = {'name': task_id, 'logdir': output_dir}
        _logger_kwargs.update(logger_kwargs or {})
        logger.init(**_logger_kwargs)

        # Basic components in Advisor.
        self.rand_prob = rand_prob
        self.optimization_strategy = optimization_strategy

        # Init the basic ingredients in Bayesian optimization.
        self.transfer_learning_history = transfer_learning_history
        self.surrogate_type = surrogate_type
        self.constraint_surrogate_type = None
        self.acq_type = acq_type
        self.acq_optimizer_type = acq_optimizer_type
        self.init_num = initial_trials
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)
        self.ref_point = ref_point

        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints, config_space=config_space,
            ref_point=ref_point, meta_info=None,  # todo: add meta info
        )

        # initial design
        if initial_configurations is not None and len(initial_configurations) > 0:
            self.initial_configurations = initial_configurations
            self.init_num = len(initial_configurations)
        else:
            self.initial_configurations = self.create_initial_design(self.init_strategy)
            self.init_num = len(self.initial_configurations)

        self.surrogate_model = None
        self.constraint_models = None
        self.acquisition_function = None
        self.optimizer = None
        self.auto_alter_model = False
        self.algo_auto_selection()
        self.check_setup()
        self.setup_bo_basics()

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

        if num_config_evaluated == 300:
            if self.surrogate_type == 'gp':
                self.surrogate_type = 'prf'
                logger.info('n_observations=300, change surrogate model from GP to PRF!')
                if self.acq_optimizer_type == 'random_scipy':
                    self.acq_optimizer_type = 'local_random'
                    logger.info('n_observations=300, change acq optimizer from random_scipy to local_random!')
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
            surrogate_str = self.surrogate_type.split('_')
            assert len(surrogate_str) == 3 and surrogate_str[0] == 'tlbo'
            assert surrogate_str[1] in ['rgpe', 'sgpr', 'topov3']  # todo: 'mfgpe'

    def setup_bo_basics(self):
        """
        Prepare the basic BO components.
        Returns
        -------
        An optimizer object.
        """
        if self.num_objectives == 1 or self.acq_type == 'parego':
            self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                                   config_space=self.config_space,
                                                   rng=self.rng,
                                                   transfer_learning_history=self.transfer_learning_history)
        else:  # multi-objectives
            self.surrogate_model = [build_surrogate(func_str=self.surrogate_type,
                                                    config_space=self.config_space,
                                                    rng=self.rng,
                                                    transfer_learning_history=self.transfer_learning_history)
                                    for _ in range(self.num_objectives)]

        if self.num_constraints > 0:
            self.constraint_models = [build_surrogate(func_str=self.constraint_surrogate_type,
                                                      config_space=self.config_space,
                                                      rng=self.rng) for _ in range(self.num_constraints)]

        if self.acq_type in ['mesmo', 'mesmoc', 'mesmoc2', 'usemo']:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       config_space=self.config_space)
        else:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       ref_point=self.ref_point)
        if self.acq_type == 'usemo':
            self.acq_optimizer_type = 'usemo_optimizer'
        self.optimizer = build_optimizer(func_str=self.acq_optimizer_type,
                                         acq_func=self.acquisition_function,
                                         config_space=self.config_space,
                                         rng=self.rng)

    def create_initial_design(self, init_strategy='default'):
        """
        Create several configurations as initial design.
        Parameters
        ----------
        init_strategy: str

        Returns
        -------
        Initial configurations.
        """
        default_config = self.config_space.get_default_configuration()
        num_random_config = self.init_num - 1
        if init_strategy == 'random':
            initial_configs = self.sample_random_configs(self.init_num)
        elif init_strategy == 'default':
            initial_configs = [default_config] + self.sample_random_configs(num_random_config)
        elif init_strategy == 'random_explore_first':
            candidate_configs = self.sample_random_configs(100)
            initial_configs = self.max_min_distance(default_config, candidate_configs, num_random_config)
        elif init_strategy == 'sobol':
            sobol = SobolSampler(self.config_space, num_random_config, random_state=self.rng)
            initial_configs = [default_config] + sobol.generate(return_config=True)
        elif init_strategy == 'latin_hypercube':
            lhs = LatinHypercubeSampler(self.config_space, num_random_config, criterion='maximin')
            initial_configs = [default_config] + lhs.generate(return_config=True)
        elif init_strategy == 'halton':
            halton = HaltonSampler(self.config_space, num_random_config, random_state=self.rng)
            initial_configs = [default_config] + halton.generate(return_config=True)
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

        valid_configs = []
        for config in initial_configs:
            try:
                config.is_valid_configuration()
            except ValueError:
                continue
            valid_configs.append(config)
        if len(valid_configs) != len(initial_configs):
            logger.warning('Only %d/%d valid configurations are generated for initial design strategy: %s. '
                                'Add more random configurations.'
                                % (len(valid_configs), len(initial_configs), init_strategy))
            num_random_config = self.init_num - len(valid_configs)
            valid_configs += self.sample_random_configs(num_random_config, excluded_configs=valid_configs)
        return valid_configs

    def max_min_distance(self, default_config, src_configs, num):
        min_dis = list()
        initial_configs = list()
        initial_configs.append(default_config)

        for config in src_configs:
            dis = np.linalg.norm(config.get_array() - default_config.get_array())
            min_dis.append(dis)
        min_dis = np.array(min_dis)

        for i in range(num):
            furthest_config = src_configs[np.argmax(min_dis)]
            initial_configs.append(furthest_config)
            min_dis[np.argmax(min_dis)] = -1

            for j in range(len(src_configs)):
                if src_configs[j] in initial_configs:
                    continue
                updated_dis = np.linalg.norm(src_configs[j].get_array() - furthest_config.get_array())
                min_dis[j] = min(updated_dis, min_dis[j])

        return initial_configs

    def get_suggestion(self, history: History = None, return_list: bool = False):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history is None:
            history = self.history

        self.alter_model(history)

        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()

        if num_config_evaluated < self.init_num:
            res = self.initial_configurations[num_config_evaluated]
            return [res] if return_list else res
        if self.optimization_strategy == 'random':
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        if (not return_list) and self.rng.random() < self.rand_prob:
            logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        X = history.get_config_array(transform='scale')
        Y = history.get_objectives(transform='infeasible')
        cY = history.get_constraints(transform='bilog')

        if self.optimization_strategy == 'bo':
            if num_config_successful < max(self.init_num, 1):
                logger.warning('No enough successful initial trials! Sample random configuration.')
                res = self.sample_random_configs(1, history)[0]
                return [res] if return_list else res

            # train surrogate model
            if self.num_objectives == 1:
                self.surrogate_model.train(X, Y[:, 0])
            elif self.acq_type == 'parego':
                weights = self.rng.random_sample(self.num_objectives)
                weights = weights / np.sum(weights)
                scalarized_obj = get_chebyshev_scalarization(weights, Y)
                self.surrogate_model.train(X, scalarized_obj(Y))
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
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     eta=scalarized_obj(np.atleast_2d(mo_incumbent_values)),
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

            # optimize acquisition function
            challengers = self.optimizer.maximize(runhistory=history,
                                                  num_points=5000)
            if return_list:
                # Caution: return_list doesn't contain random configs sampled according to rand_prob
                return challengers.challengers

            for config in challengers.challengers:
                if config not in history.configurations:
                    return config
            logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                                'Sample random config.' % (len(challengers.challengers), ))
            return self.sample_random_configs(1, history)[0]
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

    def update_observation(self, observation: Observation):
        """
        Update the current observations.
        Parameters
        ----------
        observation

        Returns
        -------

        """
        return self.history.update_observation(observation)

    def sample_random_configs(self, num_configs=1, history=None, excluded_configs=None):
        """
        Sample a batch of random configurations.
        Parameters
        ----------
        num_configs

        history

        Returns
        -------

        """
        if history is None:
            history = self.history
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in (history.configurations + configs) and config not in excluded_configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs

    def get_history(self):
        return self.history

    def save_json(self, filename: str = None):
        """
        Save history to a json file.
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f'history/{self.task_id}/history_{self.timestamp}.json')
        self.history.save_json(filename)

    def load_json(self, filename: str):
        """
        Load history from a json file.
        """
        self.history = History.load_json(filename, self.config_space)

    def get_suggestions(self):
        raise NotImplementedError
