import random
from typing import Optional, Callable, List, Type, Union

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.acquisition_function import AbstractAcquisitionFunction
from openbox.core.base import build_acq_func, build_surrogate
from openbox.core.ea.base_ea_advisor import Individual
from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.multi_objective import NondominatedPartitioning, get_chebyshev_scalarization
from openbox.utils.util_funcs import deprecate_kwarg
from openbox.utils.history import Observation


class SAEAAdvisor(ModularEAAdvisor):

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space: ConfigurationSpace,
                 num_objectives=1,
                 num_constraints=0,
                 population_size=None,
                 optimization_strategy='ea',
                 batch_size=10,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,

                 required_evaluation_count: Optional[int] = 20,
                 auto_step=True,
                 strict_auto_step=True,
                 skip_gen_population=False,
                 filter_gen_population: Optional[Callable[[List[Configuration]], List[Configuration]]] = None,
                 keep_unexpected_population=True,
                 save_cached_configuration=True,

                 ea: Union[ModularEAAdvisor, Type] = RegularizedEAAdvisor,
                 surrogate: str = 'gp_rbf',
                 constraint_surrogate: str = 'gp_rbf',
                 acq: str = None,

                 gen_multiplier=50,

                 ref_point=None
                 ):

        self.ea = ea if isinstance(ea, ModularEAAdvisor) else ea(config_space)
        population_size = population_size or self.ea.population_size
        required_evaluation_count = required_evaluation_count or self.ea.required_evaluation_count

        super().__init__(config_space=config_space, num_objectives=num_objectives, num_constraints=num_constraints,
                         population_size=population_size, optimization_strategy=optimization_strategy,
                         batch_size=batch_size, output_dir=output_dir, task_id=task_id,
                         random_state=random_state,

                         required_evaluation_count=required_evaluation_count, auto_step=auto_step,
                         strict_auto_step=strict_auto_step, skip_gen_population=skip_gen_population,
                         filter_gen_population=filter_gen_population,
                         keep_unexpected_population=keep_unexpected_population,
                         save_cached_configuration=save_cached_configuration
                         )

        # Default Acq
        acq = acq or (('ehvi' if self.num_constraints > 0 else 'ehvic') if self.num_objectives > 1 and ref_point
                      else ('mesmo' if self.num_constraints > 0 else 'mesmoc2') if self.num_objectives > 1
        else ('eic' if self.num_constraints > 0 else 'ei'))

        constraint_surrogate = constraint_surrogate or surrogate

        # TODO Compatibility check (mesmo requires gp_rbf, etc.)

        self.ea.auto_step = False

        # This is ALWAYS a list no matter multi-obj or single-obj
        self.objective_surrogates: List[AbstractModel] = [
            build_surrogate(surrogate, config_space, self.rng or random, None)
            for x in range(self.num_objectives)]
        self.constraint_surrogates: List[AbstractModel] = [build_surrogate(constraint_surrogate, config_space,
                                                                           self.rng or random, None) for x in
                                                           range(self.num_constraints)]

        mo_acq = acq in ['ehvi', 'mesmo', 'usemo', 'parego', 'ehvic', 'mesmoc', 'mesmoc2']

        # Code copied from generic_advisor.py
        # ehvi needs an extra ref_point arg.
        if acq in {'ehvi', 'ehvic'} and ref_point:
            self.ref_point = ref_point
            self.acq: AbstractAcquisitionFunction = \
                build_acq_func(acq, self.objective_surrogates if mo_acq else self.objective_surrogates[0],
                               self.constraint_surrogates, ref_point=ref_point)
        elif acq in {'ehvi', 'ehvic'} and not ref_point:
            raise ValueError('Must provide reference point to use EHVI method!')
        else:
            self.acq: AbstractAcquisitionFunction = \
                build_acq_func(acq, self.objective_surrogates if mo_acq else self.objective_surrogates[0],
                               self.constraint_surrogates, config_space=config_space)
        self.acq_type = acq

        self.gen_multiplier = gen_multiplier

        self.is_models_trained = False

    def _gen(self, count=1) -> List[Configuration]:
        if not self.is_models_trained:  # All models should be trained.
            return self.ea.get_suggestions(count)

        configs = self.ea.get_suggestions(count * self.gen_multiplier)
        results = self.acq(configs)

        res = list(zip(configs, results))
        res.sort(key=lambda x: x[1], reverse=True)

        self.ea.remove_uneval([x[0] for x in res[count:]])

        return [x[0] for x in res[:count]]

    def update_observations(self, observations: List[Observation]):
        self.ea.update_observations(observations)
        super().update_observations(observations)

    def _sel(self, parent: List[Individual], sub: List[Individual]) -> List[Individual]:
        self.ea.sel()

        X = self.history.get_config_array(transform='scale')
        Y = self.history.get_objectives(transform='infeasible')
        cY = self.history.get_constraints(transform='bilog')

        # Alternate option: use untransformed objectives (maybe enabled in the future)
        # Y = self.history.get_objectives(transform='failed')

        self.lastX = X
        self.lastY = Y

        for i in range(self.num_objectives):
            self.objective_surrogates[i].train(X, Y[:, i] if Y.ndim == 2 else Y)

        for i in range(self.num_constraints):
            self.constraint_surrogates[i].train(X, cY[:, i])

        # Code copied from generic_advisor.py

        num_config_evaluated = len(self.history)
        num_config_successful = self.history.get_success_count()
        # update acquisition function
        if self.num_objectives == 1:
            incumbent_value = self.history.get_incumbent_value()
            self.acq.update(model=self.objective_surrogates[0],
                            constraint_models=self.constraint_surrogates,
                            eta=incumbent_value,
                            num_data=num_config_evaluated)
        else:  # multi-objectives
            mo_incumbent_values = self.history.get_mo_incumbent_values()
            if self.acq_type == 'parego':
                weights = self.rng.random_sample(self.num_objectives)
                weights = weights / np.sum(weights)
                self.acq.update(model=self.objective_surrogates,
                                constraint_models=self.constraint_surrogates,
                                eta=get_chebyshev_scalarization(weights, Y)(np.atleast_2d(mo_incumbent_values)),
                                num_data=num_config_evaluated)
            elif self.acq_type.startswith('ehvi'):
                partitioning = NondominatedPartitioning(self.num_objectives, Y)
                cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                self.acq.update(model=self.objective_surrogates,
                                constraint_models=self.constraint_surrogates,
                                cell_lower_bounds=cell_bounds[0],
                                cell_upper_bounds=cell_bounds[1])
            else:
                self.acq.update(model=self.objective_surrogates,
                                constraint_models=self.constraint_surrogates,
                                constraint_perfs=cY,  # for MESMOC
                                eta=mo_incumbent_values,
                                num_data=num_config_evaluated,
                                X=X, Y=Y)

        self.is_models_trained = True

        return sub
