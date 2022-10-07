import random
import typing
from typing import Optional, Callable, List, Type, Union

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, CategoricalHyperparameter, \
    OrdinalHyperparameter

from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.acquisition_function import AbstractAcquisitionFunction
from openbox.core.base import build_acq_func, build_surrogate, Observation, build_optimizer
from openbox.core.ea.base_ea_advisor import Individual
from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.config_space import convert_configurations_to_array
from openbox.utils.history_container import HistoryContainer
from openbox.utils.multi_objective import NondominatedPartitioning, get_chebyshev_scalarization
from openbox.utils.util_funcs import check_random_state


class LinearMappedModel(AbstractModel):
    """
    A Linear Mapped Model is a 1-d model that uses a 1-d subspace results of an n-d model.
    Used by acq_maximizer in LineBO
    """

    def __init__(self,
                 father: AbstractModel,
                 x0: np.ndarray,
                 x1: np.ndarray,
                 ):
        """
        father: The original model.

        x0, x1: define the subspace and scale.

        Let's say f is this model, and F is the father model. Then we have:

        f(0) = F(x0)
        f(1) = F(x1)

        f(t) = F(x0 + t(x1 - x0))
        """
        self.father = father

        # I don't know how I should set the last 2 args. However, they're not used in LineBO.
        super().__init__(np.array([0.0]), [(0.0, 1.0)], father.instance_features, father.pca_components)

        self.x0 = x0
        self.t = x1 - x0

    def _train(self, X: np.ndarray, Y: np.ndarray) -> 'AbstractModel':
        """
        Normally, don't train this model. Train the father model directly.
        """
        self.father.train(self.x0 + self.t * X, Y)
        return self

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.father.predict(self.x0 + self.t * X)


class LineBOAdvisor:

    def __init__(self, config_space: ConfigurationSpace,
                 num_objs=1,
                 num_constraints=0,
                 task_id='default_task_id',
                 batch_size=10,
                 random_state=None,

                 surrogate: str = 'gp_rbf',
                 constraint_surrogate: str = 'gp_rbf',
                 acq: str = None,
                 acq_optimizer: str = 'random_scipy',

                 direction_strategy: str = 'random',
                 direction_switch_interval=5
                 ):
        self.num_objs = num_objs
        # Does not support multi-obj!
        assert self.num_objs == 1
        # Supports one or more constraints
        self.num_constraints = num_constraints

        self.config_space = config_space
        self.dim = len(config_space.keys())
        self.batch_size = batch_size
        self.rng = check_random_state(random_state)
        self.task_id = task_id

        acq = acq or ('eic' if self.num_constraints > 0 else 'ei')
        self.acq_type = acq
        self.acq_optimizer_type = acq_optimizer

        constraint_surrogate = constraint_surrogate or surrogate

        self.objective_surrogate: AbstractModel = build_surrogate(surrogate, config_space, self.rng or random, None)
        self.constraint_surrogates: List[AbstractModel] = [build_surrogate(constraint_surrogate, config_space,
                                                                           self.rng or random, None) for x in
                                                           range(self.num_constraints)]

        self.line_space = ConfigurationSpace()
        self.line_space.add_hyperparameters([UniformFloatHyperparameter('x', 0, 1)])

        assert direction_strategy in ['random', 'aligned', 'descent']
        self.direction_strategy = direction_strategy
        self.direction_switch_interval = direction_switch_interval

        self.current_subspace: Optional[Union[np.ndarray, np.ndarray]] = None
        self.acq: Optional[AbstractAcquisitionFunction] = None
        self.acq_optimizer = None

        self.history_container = HistoryContainer(task_id, num_constraints, self.config_space)

        self.cnt = 0

    def update_subspace(self):
        if self.direction_strategy == 'random':
            d = self.rng.randn(self.dim)
            d = d / np.linalg.norm(d)
            direction = d
        elif self.direction_strategy == 'aligned':
            direction = np.zeros(self.dim)
            direction[self.rng.randint(0, self.dim - 1)] = 1
        else:
            direction = np.zeros(self.dim)

        incumbent = self.config_space.sample_configuration() if len(self.history_container.incumbents) == 0 else \
            self.history_container.incumbents[-1][0]

        x = incumbent.get_array()

        mx = 1e100
        mn = 1e100

        for i, key in enumerate(self.config_space.keys()):

            scale = 1

            hp_type = self.config_space.get_hyperparameter(key)
            if isinstance(hp_type, CategoricalHyperparameter) or isinstance(hp_type, OrdinalHyperparameter):
                scale = hp_type.get_size()
                direction[i] *= hp_type.get_size()

            if direction[i] == 0:
                pass
            elif direction[i] > 0:
                mx = min(mx, (scale - x[i]) / direction[i])
                mn = min(mn, (x[i] - 0) / direction[i])
            else:
                mx = min(mx, (0 - x[i]) / direction[i])
                mn = min(mn, (x[i] - scale) / direction[i])

        x0 = x - mn * direction
        x1 = x + x * mx * direction
        self.current_subspace = (x0, x1)

        self.acq = build_acq_func(self.acq_type,
                                  LinearMappedModel(self.objective_surrogate, x0, x1),
                                  [LinearMappedModel(i, x0, x1) for i in self.constraint_surrogates],
                                  config_space=self.line_space)
        self.acq_optimizer = build_optimizer(func_str=self.acq_optimizer_type,
                                             acq_func=self.acq,
                                             config_space=self.line_space,
                                             rng=self.rng)

    def to_original_space(self, X):
        if isinstance(X, Configuration):
            X = X.get_array()

        oX = self.current_subspace[0] + (self.current_subspace[1] - self.current_subspace[0]) * X
        return Configuration(self.config_space, vector=oX)

    def get_suggestion(self):
        if len(self.history_container.configurations) == 0:
            return self.config_space.sample_configuration()

        if self.cnt % self.direction_switch_interval == 0 or self.current_subspace is None:
            self.update_subspace()

        self.cnt += 1

        incumbent_value = self.history_container.get_incumbents()[0][1]
        num_config_evaluated = len(self.history_container.configurations)
        self.acq.update(eta=incumbent_value,
                        num_data=num_config_evaluated)

        X = convert_configurations_to_array(self.history_container.configurations)
        Y = self.history_container.get_transformed_perfs(transform=None)
        cY = self.history_container.get_transformed_constraint_perfs(transform='bilog')

        self.objective_surrogate.train(X, Y[:, 0] if Y.ndim == 2 else Y)
        for i in range(self.num_constraints):
            self.constraint_surrogates[i].train(X, cY[:, i])

        challengers = self.acq_optimizer.maximize(runhistory=HistoryContainer(task_id=self.task_id,
                                                                              num_constraints=self.num_constraints,
                                                                              config_space=self.line_space),
                                                  num_points=5000)

        ret = None

        for config in challengers.challengers:
            if config not in self.history_container.configurations:
                ret = config
                break

        if ret is None:
            return self.to_original_space(self.line_space.sample_configuration())
        else:
            return self.to_original_space(ret)

    def update_observation(self, observation: Observation):
        self.history_container.update_observation(observation)

        incumbent_value = self.history_container.get_incumbents()[0][1]
        num_config_evaluated = len(self.history_container.configurations)
        self.acq.update(eta=incumbent_value,
                        num_data=num_config_evaluated)

    def get_suggestions(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        return [self.get_suggestion() for i in range(batch_size)]

    def update_observations(self, observations: List[Observation]):
        return [self.update_observation(o) for o in observations]

    def get_history(self):
        return self.history_container
