import random

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter

from typing import *

from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox.core.ea.regularized_ea_advisor import Observation, RegularizedEAAdvisor
from openbox.acq_maximizer.ei_optimization import AcquisitionFunctionMaximizer
from openbox.acquisition_function import EI, AbstractAcquisitionFunction, EIC
from openbox.core.base import build_acq_func, build_surrogate

from openbox.core.ea.base_ea_advisor import *
from openbox.core.ea.base_modular_ea_advisor import *
from openbox.core.ea.cmaes_ea_advisor import CMAESEAAdvisor
from openbox.surrogate.base.base_model import AbstractModel
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.surrogate.base.gp import GaussianProcess
from openbox.utils.config_space import convert_configurations_to_array
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.multi_objective import NondominatedPartitioning, get_chebyshev_scalarization


class SAEA_Advisor(ModularEAAdvisor):

    def __init__(self, config_space: ConfigurationSpace,
                 num_objs = 1,
                 num_constraints = 0,
                 population_size = None,
                 optimization_strategy = 'ea',
                 batch_size = 10,
                 output_dir = 'logs',
                 task_id = 'default_task_id',
                 random_state = None,

                 required_evaluation_count: Optional[int] = 20,
                 auto_step = True,
                 strict_auto_step = True,
                 skip_gen_population = False,
                 filter_gen_population: Optional[Callable[[List[Configuration]], List[Configuration]]] = None,
                 keep_unexpected_population = True,
                 save_cached_configuration = True,

                 ea: Union[ModularEAAdvisor, Type] = RegularizedEAAdvisor,
                 surrogate: str = 'gp_rbf',
                 constraint_surrogate: str = 'gp_rbf',
                 acq: str = None,

                 gen_multiplier = 50,

                 **kwargs
                 ):

        # assert num_objs == 1

        self.ea = ea if isinstance(ea, ModularEAAdvisor) else ea(config_space)
        population_size = population_size or self.ea.population_size
        required_evaluation_count = required_evaluation_count or self.ea.required_evaluation_count

        ModularEAAdvisor.__init__(self, config_space = config_space, num_objs = num_objs,
                                  num_constraints = num_constraints,
                                  population_size = population_size, optimization_strategy = optimization_strategy,
                                  batch_size = batch_size, output_dir = output_dir, task_id = task_id,
                                  random_state = random_state,

                                  required_evaluation_count = required_evaluation_count, auto_step = auto_step,
                                  strict_auto_step = strict_auto_step, skip_gen_population = skip_gen_population,
                                  filter_gen_population = filter_gen_population,
                                  keep_unexpected_population = keep_unexpected_population,
                                  save_cached_configuration = save_cached_configuration
                                  )

        # Default Acq
        acq = acq or ('mesmo' if self.num_constraints > 0 else 'mesmoc2') if self.num_objs > 1 else \
            ('eic' if self.num_constraints > 0 else 'ei')

        constraint_surrogate = constraint_surrogate or surrogate

        # TODO Compatibility check (mesmo requires gp_rbf, etc.)

        self.ea.auto_step = False

        # This is ALWAYS a list no matter multi-obj or single-obj
        self.objective_surrogates: List[AbstractModel] = [
            build_surrogate(surrogate, config_space, self.rng or random, None)
            for x in range(self.num_objs)]
        self.constraint_surrogates: List[AbstractModel] = [build_surrogate(constraint_surrogate, config_space,
                                                                           self.rng or random, None) for x in
                                                           range(self.num_constraints)]

        mo_acq = acq in ['ehvi', 'mesmo', 'usemo', 'parego', 'ehvic', 'mesmoc', 'mesmoc2']

        # Code copied from generic_advisor.py
        # ehvi needs an extra ref_point arg.
        if acq == 'ehvi' and 'ref_point' in kwargs:
            self.ref_point = kwargs['ref_point']
            self.acq: AbstractAcquisitionFunction = \
                build_acq_func(acq, self.objective_surrogates if mo_acq else self.objective_surrogates[0],
                               self.constraint_surrogates, ref_point = kwargs['ref_point'])
        elif acq == 'ehvi' and 'ref_point' not in kwargs:
            raise ValueError(
                'Must provide reference point to use EHVI method! (Add ref_point=... to constructor kwargs)')
        else:
            self.acq: AbstractAcquisitionFunction = \
                build_acq_func(acq, self.objective_surrogates if mo_acq else self.objective_surrogates[0],
                               self.constraint_surrogates, config_space = config_space)
        self.acq_type = acq

        self.gen_multiplier = gen_multiplier

    def _gen(self, count = 1) -> List[Configuration]:
        if [x for x in self.objective_surrogates if not x.is_trained]:  # All models should be trained.
            l = self.ea.get_suggestions(count)
            return l

        configs = self.ea.get_suggestions(count * self.gen_multiplier)
        results = self.acq(configs)

        res = list(zip(configs, results))
        res.sort(key = lambda x: x[1], reverse = True)

        self.ea.remove_uneval([x[0] for x in res[count:]])

        return [x[0] for x in res[:count]]

    def update_observations(self, observations: List[Observation]):
        self.ea.update_observations(observations)
        super().update_observations(observations)

    def _sel(self, parent: List[Individual], sub: List[Individual]) -> List[Individual]:
        self.ea.sel()

        X = convert_configurations_to_array(self.history_container.configurations)
        Y = self.history_container.get_transformed_perfs(transform = None)

        # Alternate option: use untransformed perfs (may be enabled in the future)
        # Y = np.array(self.history_container.perfs)

        self.lastX = X
        self.lastY = Y

        cY = self.history_container.get_transformed_constraint_perfs(transform = 'bilog')

        # ok_idx = np.min((cY < 0), axis=1)

        for i in range(self.num_objs):
            self.objective_surrogates[i].train(X, Y[:, i] if Y.ndim == 2 else Y)
            # self.objective_surrogates.train(X[ok_idx], Y[ok_idx, i] if Y.ndim == 2 else Y)

        for i in range(self.num_constraints):
            self.constraint_surrogates[i].train(X, cY[:, i])

        # Code copied from generic_advisor.py

        num_config_evaluated = len(self.history_container.configurations)
        num_config_successful = len(self.history_container.successful_perfs)
        # update acquisition function
        if self.num_objs == 1:
            incumbent_value = self.history_container.get_incumbents()[0][1]
            self.acq.update(model = self.objective_surrogates[0],
                            constraint_models = self.constraint_surrogates,
                            eta = incumbent_value,
                            num_data = num_config_evaluated)
        else:  # multi-objectives
            mo_incumbent_value = self.history_container.get_mo_incumbent_value()
            if self.acq_type == 'parego':
                weights = self.rng.random_sample(self.num_objs)
                weights = weights / np.sum(weights)
                self.acq.update(model = self.objective_surrogates,
                                constraint_models = self.constraint_surrogates,
                                eta = get_chebyshev_scalarization(weights, Y)(np.atleast_2d(mo_incumbent_value)),
                                num_data = num_config_evaluated)
            elif self.acq_type.startswith('ehvi'):
                partitioning = NondominatedPartitioning(self.num_objs, Y)
                cell_bounds = partitioning.get_hypercell_bounds(ref_point = self.ref_point)
                self.acq.update(model = self.objective_surrogates,
                                constraint_models = self.constraint_surrogates,
                                cell_lower_bounds = cell_bounds[0],
                                cell_upper_bounds = cell_bounds[1])
            else:
                self.acq.update(model = self.objective_surrogates,
                                constraint_models = self.constraint_surrogates,
                                constraint_perfs = cY,  # for MESMOC
                                eta = mo_incumbent_value,
                                num_data = num_config_evaluated,
                                X = X, Y = Y)

        return sub
