import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from openbox.benchmark.objective_functions.synthetic import BaseTestProblem


class Schwefel(BaseTestProblem):
    r"""Generalized Schwefel's Problem 2.26.

    d-dimensional function (usually evaluated on `[-500, 500]^d`):

        f(x) = \sum_{i=1}^{d} -x_i sin(\sqrt{\abs{x_i}})

    """

    def __init__(self, dim=2, noise_std=0, random_state=None):
        self.dim = dim
        params = {'x%d' % i: (-500.0, 500.0, 100.0) for i in range(1, 1 + self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         optimal_value=-12596.5,
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        result['objs'] = [np.sum(-X * np.sin(np.sqrt(np.abs(X))), axis=-1)]
        return result


class Rastrigin(BaseTestProblem):
    r"""Generalized Rastrigin's Function

    d-dimensional function (usually evaluated on `[-5.12, 5.12]^d`):

        f(x) = \sum_{i=1}^{d} (x_i^2 - 10 \cos(2\pi x) + 10)

    """

    def __init__(self, dim=2, noise_std=0, random_state=None):
        self.dim = dim
        params = {'x%d' % i: (-5.12, 5.12, 1) for i in range(1, 1 + self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         optimal_value=0,
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        result['objs'] = [np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=-1)]
        return result


class Griewank(BaseTestProblem):
    r"""Generalized Griewank's Function

    d-dimensional function (usually evaluated on `[-600, 600]^d`):

        f(x) = 1/4000 \sum_{i=1}^{d} - \prod_{i=1}^d \cos(x_i / \sqrt{i}) + 1

    """

    def __init__(self, dim=2, noise_std=0, random_state=None):
        self.dim = dim
        params = {'x%d' % i: (-600, 600, 100) for i in range(1, 1 + self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         optimal_value=0,
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        result['objs'] = [np.sum(X ** 2, axis=-1) / 4000.0 -
                          np.prod(np.cos(X / (np.arange(1, self.dim - 1)) ** 0.5)) + 1]
        return result
