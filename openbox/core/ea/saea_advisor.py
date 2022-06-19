import random

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter

from typing import *

from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox import Observation
from openbox.acq_maximizer.ei_optimization import AcquisitionFunctionMaximizer
from openbox.acquisition_function import EI, AbstractAcquisitionFunction

from openbox.core.ea.base_ea_advisor import *
from openbox.core.ea.base_modular_ea_advisor import *
from openbox.core.ea.cmaes_ea_advisor import CMAESEAAdvisor
from openbox.surrogate.base.base_model import AbstractModel
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.surrogate.base.gp import GaussianProcess
from openbox.utils.constants import MAXINT, SUCCESS


class SAEA_Advisor(ModularEAAdvisor):


    def __init__(self, config_space: ConfigurationSpace,
                 num_objs=1,
                 num_constraints=0,
                 population_size=None,
                 optimization_strategy='ea',
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None,

                 required_evaluation_count: Optional[int] = None,
                 auto_step=True,
                 strict_auto_step=True,
                 skip_gen_population=False,
                 filter_gen_population: Optional[Callable[[List[Configuration]], List[Configuration]]] = None,
                 keep_unexpected_population=True,
                 save_cached_configuration=True,

                 force_blackbox_constraints = True,

                 ea: Union[ModularEAAdvisor, Type] = CMAESEAAdvisor,
                 objective_surrogate: Union[AbstractModel, str, Callable[[], AbstractModel]] = 'gp',
                 objective_acq: Union[AbstractAcquisitionFunction, Type] = EI,
                 constraint_surrogate: Union[AbstractModel, str, Callable[[], AbstractModel]] = 'gp',
                 constraint_acq: Union[AbstractAcquisitionFunction, Type] = EI,

                 gen_multiplier = 1000
                 ):

        self.ea = ea if isinstance(ea, ModularEAAdvisor) else ea(config_space)
        population_size = population_size or self.ea.population_size

        ModularEAAdvisor.__init__(self, config_space=config_space, num_objs=num_objs, num_constraints=num_constraints,
                                  population_size=population_size, optimization_strategy=optimization_strategy,
                                  batch_size=batch_size, output_dir=output_dir, task_id=task_id,
                                  random_state=random_state,

                                  required_evaluation_count=required_evaluation_count, auto_step=auto_step,
                                  strict_auto_step=strict_auto_step, skip_gen_population=skip_gen_population,
                                  filter_gen_population=filter_gen_population,
                                  keep_unexpected_population=keep_unexpected_population,
                                  save_cached_configuration=save_cached_configuration
                                  )

        self.force_blackbox_constraints = force_blackbox_constraints

        ea.auto_step = False

        config_types = []
        bounds = []
        for i in config_space.keys():
            x = config_space.get_hyperparameter(i)
            if isinstance(x, NumericalHyperparameter):
                config_types.append(0)
                bounds.append((x.lower, x.upper))
            else:
                config_types.append(x.get_size())
                bounds.append((x.get_size(), np.nan))

        def conv_sur(sur: Union[AbstractModel, str, Callable[[], AbstractModel]]):
            if isinstance(sur, AbstractModel):
                return sur
            elif isinstance(sur, str):
                return create_gp_model(sur, config_space, config_types, bounds, random_state or self.rng)
            elif callable(sur):
                return sur()

        def conv_acq(acq: Union[AbstractAcquisitionFunction, Type], surrogate):
            if isinstance(acq, AbstractAcquisitionFunction):
                return acq
            elif type(acq) == type:
                return acq(surrogate)

        self.objective_surrogate: AbstractModel = conv_sur(objective_surrogate)
        self.objective_acq: AbstractAcquisitionFunction = conv_acq(objective_acq, self.objective_surrogate)

        if num_constraints > 0:
            self.constraint_surrogate: AbstractModel = conv_sur(constraint_surrogate)
            self.constraint_acq: AbstractAcquisitionFunction = conv_acq(constraint_acq, self.constraint_surrogate)
        else:
            self.constraint_surrogate = None
            self.constraint_acq = None

        self.gen_multiplier = gen_multiplier

        self.all_history: List[Individual] = []

    def _gen(self, count=1) -> List[Configuration]:
        if not self.objective_surrogate.is_trained:
            return self.ea.get_suggestions(count)

        configs = self.ea.get_suggestions(count * self.gen_multiplier)
        results = self.objective_acq(configs)

        if self.constraint_surrogate:
            results1 = self.constraint_acq(configs)
            indivs = [Individual(c, r, constraint_check(r1)) for c, r, r1 in zip(configs, results, results1)]
            indivs = pareto_sort(indivs)

            a = [x for x in indivs if x.constraints_satisfied]
            b = [x for x in indivs if not x.constraints_satisfied]
            indivs = a + b

        else:
            indivs = [Individual(c, r) for c, r in zip(configs, results)]
            indivs = pareto_sort(indivs)

        self.ea.remove_uneval([x.config for x in indivs[count:]])

        return [x.config for x in indivs[:count]]


    def update_observations(self, observations: List[Observation]):
        self.ea.update_observations(observations)
        self.all_history.extend([as_individual(x) for x in observations])
        super().update_observations(observations)

    def _sel(self, parent: List[Individual], sub: List[Individual]) -> List[Individual]:

        self.ea.sel()

        X = np.stack([x.config for x in self.all_history])

        self.objective_surrogate.train(X, np.stack([x.perf for x in self.all_history]))

        if self.constraint_surrogate:
            self.constraint_surrogate.train(X, np.stack([-1 if x.constraints_satisfied else 1 for x in self.all_history]))

        return sub

