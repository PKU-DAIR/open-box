# License: MIT

import numpy as np

from openbox import logger
from openbox.core.base import build_acq_func, build_optimizer, build_surrogate
from openbox.core.generic_advisor import Advisor
from openbox.utils.history import Observation, MultiStartHistory
from openbox.utils.multi_objective import NondominatedPartitioning
from openbox.utils.trust_region import TurboState
from openbox.utils.util_funcs import get_types, deprecate_kwarg


class MCAdvisor(Advisor):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            mc_times=10,
            initial_trials=3,
            initial_configurations=None,
            init_strategy='random_explore_first',
            transfer_learning_history=None,
            optimization_strategy='bo',
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            use_trust_region=False,
            ref_point=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
    ):

        self.mc_times = mc_times
        self.use_trust_region = use_trust_region
        self.turbo_state = None

        super().__init__(config_space,
                         num_objectives=num_objectives,
                         num_constraints=num_constraints,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         init_strategy=init_strategy,
                         transfer_learning_history=transfer_learning_history,
                         optimization_strategy=optimization_strategy,
                         surrogate_type=surrogate_type,
                         acq_type=acq_type,
                         acq_optimizer_type=acq_optimizer_type,
                         ref_point=ref_point,
                         output_dir=output_dir,
                         task_id=task_id,
                         random_state=random_state,
                         logger_kwargs=logger_kwargs)

        if self.use_trust_region:
            self.history = MultiStartHistory(
                task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints,
                config_space=config_space, ref_point=ref_point, meta_info=None,  # todo: add meta info
            )

    def algo_auto_selection(self):
        info_str = ''

        if self.surrogate_type == 'auto':
            self.surrogate_type = 'gp'
            info_str += ' surrogate_type: %s.' % self.surrogate_type

        if self.acq_type == 'auto':
            if self.num_objectives == 1:  # single objective
                if self.num_constraints == 0:
                    self.acq_type = 'mcei'
                else:  # with constraints
                    self.acq_type = 'mceic'
            elif self.num_objectives <= 4:  # multi objective (<=4)
                if self.num_constraints == 0:
                    self.acq_type = 'mcehvi'
                else:  # with constraints
                    self.acq_type = 'mcehvic'
            else:  # multi objective (>4)
                if self.num_constraints == 0:
                    self.acq_type = 'mcparego'
                else:  # with constraints
                    self.acq_type = 'mcparegoc'
            info_str += ' acq_type: %s.' % self.acq_type

        if self.acq_optimizer_type == 'auto':
            self.acq_optimizer_type = 'batchmc'
            info_str += ' acq_optimizer_type: %s.' % self.acq_optimizer_type

        if info_str != '':
            info_str = '[BO auto selection] ' + info_str
            logger.info(info_str)

    def check_setup(self):
        """
            check num_objectives, num_constraints, acq_type, surrogate_type
        """
        assert isinstance(self.num_objectives, int) and self.num_objectives >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

        if self.surrogate_type is None:
            self.surrogate_type = 'gp'
        assert self.surrogate_type in ['gp', ]  # MC sample method
        if self.constraint_surrogate_type is None:
            self.constraint_surrogate_type = 'gp'

        # Single objective
        if self.num_objectives == 1:
            if self.num_constraints == 0:
                assert self.acq_type in ['mcei']
            else:
                assert self.acq_type in ['mceic'] or self.use_trust_region

        # Multi objective
        else:
            if self.num_constraints == 0:
                assert self.acq_type in ['mcparego', 'mcehvi']
            else:
                assert self.acq_type in ['mcparegoc', 'mcehvic'] or self.use_trust_region

            # Check reference point is provided for EHVI methods
            if 'ehvi' in self.acq_type and self.ref_point is None:
                raise ValueError('Must provide reference point to use EHVI method!')

    def setup_bo_basics(self):
        if self.num_objectives == 1:
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

        self.acquisition_function = build_acq_func(func_str=self.acq_type, model=self.surrogate_model,
                                                   constraint_models=self.constraint_models,
                                                   mc_times=self.mc_times, ref_point=self.ref_point)

        self.optimizer = build_optimizer(func_str=self.acq_optimizer_type,
                                         acq_func=self.acquisition_function,
                                         config_space=self.config_space,
                                         rng=self.rng)

        if self.use_trust_region:
            types, bounds = get_types(self.config_space)
            cont_dim = np.sum(types == 0)
            self.turbo_state = TurboState(cont_dim)

    def get_suggestion(self, history=None):
        if history is None:
            history = self.history

        # Check if turbo needs to be restarted
        if self.use_trust_region and self.turbo_state.restart_triggered:
            history.restart()
            logger.info('Turbo Restart!')

        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()

        if num_config_evaluated < self.init_num:
            return self.initial_configurations[num_config_evaluated]

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(1)[0]

        X = history.get_config_array(transform='scale')
        Y = history.get_objectives(transform='infeasible')
        cY = history.get_constraints(transform='bilog')

        if self.optimization_strategy == 'bo':
            if num_config_successful < max(self.init_num, 1):
                logger.warning('No enough successful initial trials! Sample random configuration.')
                return self.sample_random_configs(1)[0]

            # train surrogate model
            if self.num_objectives == 1:
                self.surrogate_model.train(X, Y[:, 0])
            else:  # multi-objectives
                for i in range(self.num_objectives):
                    self.surrogate_model[i].train(X, Y[:, i])

            # train constraint model
            for i in range(self.num_constraints):
                self.constraint_models[i].train(X, cY[:, i])

            # update acquisition function
            if self.num_objectives == 1:  # MC-EI
                incumbent_value = history.get_incumbent_value()
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value)
            else:  # MC-ParEGO or MC-EHVI
                if self.acq_type.startswith('mcparego'):
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models)
                elif self.acq_type.startswith('mcehvi'):
                    partitioning = NondominatedPartitioning(self.num_objectives, Y)
                    cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     cell_lower_bounds=cell_bounds[0],
                                                     cell_upper_bounds=cell_bounds[1])

            # optimize acquisition function
            challengers = self.optimizer.maximize(runhistory=history,
                                                  num_points=5000,
                                                  turbo_state=self.turbo_state)
            is_repeated_config = True
            repeated_time = 0
            cur_config = None
            while is_repeated_config:
                cur_config = challengers.challengers[repeated_time]
                if cur_config in history.configurations:
                    repeated_time += 1
                else:
                    is_repeated_config = False
            return cur_config
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

    def update_observation(self, observation: Observation):
        super().update_observation(observation)
        if self.use_trust_region:
            objectives = observation.objectives
            if self.num_objectives > 1:
                raise NotImplementedError()
            else:
                self.turbo_state.update(objectives[0])
