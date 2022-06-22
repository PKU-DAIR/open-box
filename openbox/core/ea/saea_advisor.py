import random

import numpy as np
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter

from typing import *

from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox import Observation
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

                 required_evaluation_count: Optional[int] = 10,
                 auto_step = True,
                 strict_auto_step = True,
                 skip_gen_population = False,
                 filter_gen_population: Optional[Callable[[List[Configuration]], List[Configuration]]] = None,
                 keep_unexpected_population = True,
                 save_cached_configuration = True,

                 ea: Union[ModularEAAdvisor, Type] = CMAESEAAdvisor,
                 surrogate: str = 'gp',
                 acq: str = None,

                 gen_multiplier = 50
                 ):

        assert num_objs == 1

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

        acq = acq or ('eic' if self.num_constraints > 0 else 'ei')

        self.ea.auto_step = False

        self.objective_surrogates: AbstractModel = build_surrogate(surrogate, config_space, self.rng or random, None)
        self.constraint_surrogates: List[AbstractModel] = [
            build_surrogate(surrogate, config_space, self.rng or random, None)
            for x in range(self.num_constraints)]
        self.acq: AbstractAcquisitionFunction = build_acq_func(acq, self.objective_surrogates,
                                                               self.constraint_surrogates)

        self.gen_multiplier = gen_multiplier

    def _gen(self, count = 1) -> List[Configuration]:
        if not self.objective_surrogates.is_trained:
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

        # print('saea seled')

        self.ea.sel()

        X = convert_configurations_to_array(self.history_container.configurations)
        Y = self.history_container.get_transformed_perfs(transform = None)
        # Y = np.array(self.history_container.perfs)

        self.lastX = X
        self.lastY = Y

        cY = self.history_container.get_transformed_constraint_perfs(transform = 'bilog')

        # ok_idx = np.min((cY < 0), axis=1)

        for i in range(self.num_objs):
            self.objective_surrogates.train(X, Y[:, i] if Y.ndim == 2 else Y)
            # self.objective_surrogates.train(X[ok_idx], Y[ok_idx, i] if Y.ndim == 2 else Y)

        for i in range(self.num_constraints):
            self.constraint_surrogates[i].train(X, cY[:, i])

        self.acq.update(eta = Y.min())

        return sub
