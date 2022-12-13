import typing

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, UniformFloatHyperparameter, CategoricalHyperparameter, \
    OrdinalHyperparameter

from openbox.core.base import build_acq_func, build_surrogate, build_optimizer
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.history import Observation, History
from openbox.utils.util_funcs import check_random_state, get_types, deprecate_kwarg


class LinearMappedModel(AbstractModel):
    """
    A Linear Mapped Model is a 1-d model that uses a 1-d subspace results of an n-d model.
    Used by acq_maximizer in LineBO
    """

    def __init__(self,
                 father: AbstractModel,
                 x0: np.ndarray,
                 x1: np.ndarray,
                 line_space: ConfigurationSpace
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

        types, bounds = get_types(line_space)
        super().__init__(types, bounds)

        self.x0 = x0
        self.t = x1 - x0

    def _train(self, X: np.ndarray, Y: np.ndarray) -> AbstractModel:
        """
        Normally, don't train this model. Train the father model directly.
        """
        self.father.train(self.x0 + self.t * X, Y)
        return self

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.father.predict(self.x0 + self.t * X)


class LineBOAdvisor:

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space: ConfigurationSpace,
                 num_objectives=1,
                 num_constraints=0,
                 task_id='OpenBox',
                 random_state=None,

                 surrogate: str = 'gp',
                 constraint_surrogate: str = 'gp',

                 acq: str = None,
                 acq_optimizer: str = 'random_scipy',

                 direction_strategy: str = 'coordinate',

                 subbo_evals=10,
                 subbo_samples=5000,
                 ):
        self.last_gp_data = None
        self.num_objectives = num_objectives
        # Does not support multi-obj!
        assert self.num_objectives == 1
        # Supports one or more constraints
        self.num_constraints = num_constraints

        self.config_space = config_space
        self.dim = len(config_space.keys())
        self.rng = check_random_state(random_state)
        self.task_id = task_id

        acq = acq or ('eic' if self.num_constraints > 0 else 'ei')
        self.acq_type = acq
        self.acq_optimizer_type = acq_optimizer

        constraint_surrogate = constraint_surrogate or surrogate

        self.objective_surrogate = build_surrogate(surrogate, config_space, self.rng, None)
        self.constraint_surrogates = [build_surrogate(constraint_surrogate, config_space,
                                                                           self.rng, None) for x in
                                                           range(self.num_constraints)]

        self.line_space = ConfigurationSpace()
        self.line_space.add_hyperparameters([UniformFloatHyperparameter('x', 0, 1)])

        assert direction_strategy in ['random', 'coordinate', 'gradient']
        self.direction_strategy = direction_strategy
        self.subbo_evals = subbo_evals
        self.subbo_samples = subbo_samples

        self.current_subspace = None
        self.global_acq = build_acq_func(self.acq_type, self.objective_surrogate, self.constraint_surrogates,
                                         config_space=self.config_space)

        self.subspace_acq = None
        self.acq_optimizer = None

        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints, config_space=config_space,
            ref_point=None, meta_info=None,  # todo: add meta info
        )

        self.sub_history = None

        self.cnt = 0

        self.visited_incumbents = []
        self.history_lines = []

    def update_subspace(self):
        incumbent_configs = self.history.get_incumbent_configs()

        if len(incumbent_configs) == 0:
            incumbent = self.config_space.sample_configuration()
        else:
            incumbent = self.rng.choice(incumbent_configs)

        x = incumbent.get_array()

        direction_strategy = self.direction_strategy

        # if direction_strategy == 'mixed':
        #     direction_strategy = 'gradient' if self.cnt > 500 else 'coordinate'

        if direction_strategy == 'random':
            d = self.rng.randn(self.dim)
            d = d / np.linalg.norm(d)
            direction = d
        elif direction_strategy == 'coordinate':
            direction = np.zeros(self.dim)
            direction[(self.cnt // self.subbo_evals) % self.dim] = 1
        elif direction_strategy == 'gradient':
            # print("x", self.cnt)
            if self.cnt == 0:
                d = self.rng.randn(self.dim)
                d = d / np.linalg.norm(d)
                direction = d
            else:

                self.visited_incumbents.append(incumbent)

                def eval_grad(a: np.ndarray):
                    eps = np.linalg.norm(a) * 1e-4

                    toeval = [a]
                    res = np.zeros(self.dim)

                    for i in range(self.dim):
                        a1 = a.copy()
                        a1[i] += eps
                        toeval.append(a1)

                    r = self.objective_surrogate.predict(np.array(toeval))[0]

                    for i in range(self.dim):
                        res[i] = (r[i + 1] - r[0]) / eps

                    return res

                grad = eval_grad(x)
                ngrad = np.linalg.norm(grad)
                # print("grad ", grad)
                # print("|grad| ", ngrad)

                direction = grad / ngrad

        else:
            raise ValueError("Unknown Direction Strategy: " + self.direction_strategy)

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
        x1 = x + mx * direction

        self.current_subspace = (x0, x1)
        self.history_lines.append((x0, x1))

        if self.subbo_evals != 0:
            self.subspace_acq = build_acq_func(self.acq_type,
                                               LinearMappedModel(self.objective_surrogate, x0, x1, self.line_space),
                                               [LinearMappedModel(i, x0, x1, self.line_space) for i in
                                                self.constraint_surrogates],
                                               config_space=self.line_space)
            self.acq_optimizer = build_optimizer(func_str=self.acq_optimizer_type, acq_func=self.subspace_acq,
                                                 config_space=self.line_space, rng=self.rng)
            self.sub_history = History(
                task_id=self.task_id+'-sub', num_objectives=self.num_objectives, num_constraints=self.num_constraints,
                config_space=self.line_space, ref_point=None, meta_info=None,  # todo: add meta info
            )

        # print("subspace updated ", self.current_subspace)

    def to_original_space(self, X):
        if isinstance(X, Configuration):
            X = X.get_array()

        oX = self.current_subspace[0] + (self.current_subspace[1] - self.current_subspace[0]) * X.item()
        return Configuration(self.config_space, vector=oX)

    def get_suggestion(self):
        if len(self.history) == 0:
            return self.config_space.sample_configuration()

        incumbent_value = self.history.get_incumbent_value()
        num_config_evaluated = len(self.history)

        X = self.history.get_config_array(transform='scale')
        Y = self.history.get_objectives(transform='infeasible')
        cY = self.history.get_constraints(transform='bilog')

        self.last_gp_data = (X, Y)

        self.objective_surrogate.train(X, Y[:, 0] if Y.ndim == 2 else Y)

        for i in range(self.num_constraints):
            self.constraint_surrogates[i].train(X, cY[:, i])

        self.global_acq.update(eta=incumbent_value,
                               num_data=num_config_evaluated)

        interval = self.subbo_evals if self.subbo_evals != 0 else 1
        if self.cnt % interval == 0 or self.current_subspace is None:
            self.update_subspace()

        self.cnt += 1

        if self.subbo_evals != 0:

            sub_num_config_evaluated = len(self.sub_history)

            if sub_num_config_evaluated == 0:
                return self.to_original_space(self.line_space.sample_configuration())

            sub_incumbent_value = self.sub_history.get_incumbent_value()

            self.subspace_acq.update(eta=sub_incumbent_value, num_data=sub_num_config_evaluated)

            challengers = self.acq_optimizer.maximize(runhistory=self.sub_history,
                                                      num_points=self.subbo_samples)
            ret = None

            for config in challengers.challengers:
                c = self.to_original_space(config)
                cx = c.get_array()
                if any(np.linalg.norm(cx - i.get_array()) < 1e-6 for i in self.history.configurations):
                    continue

                if config not in self.history.configurations:
                    ret = c
                    break

            if ret is not None:
                return ret
            else:
                return self.to_original_space(self.line_space.sample_configuration())
        else:
            # grid_search
            gs_X = np.linspace(self.current_subspace[0], self.current_subspace[1], self.subbo_samples)
            gs_configs = [Configuration(self.config_space, vector=gs_X[i]) for i in range(self.subbo_samples)]
            res = self.objective_surrogate.predict(gs_X)[0].flatten()

            # select best of grid_search result
            ranks = np.argsort(res)

            ans = None

            for i in range(self.subbo_samples):
                if gs_configs[ranks[i]] not in self.history.configurations:
                    ans = gs_configs[ranks[i]]
                    break

            if ans is None:
                ans = self.config_space.sample_configuration()

            return ans

    def update_observation(self, observation: Observation):
        self.history.update_observation(observation)

    def get_history(self):
        return self.history
