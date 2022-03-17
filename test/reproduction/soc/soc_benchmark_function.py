import numpy as np

# from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, InCondition, EqualsCondition, UnParametrizedHyperparameter, \
    ForbiddenEqualsClause, ForbiddenInClause, ForbiddenAndConjunction


def get_problem(problem_str, **kwargs):
    # problem_str = problem_str.lower()  # dataset name may be uppercase
    if problem_str == 'townsend':
        problem = townsend
    elif problem_str == 'keane':
        problem = keane
    elif problem_str == 'ackley':
        problem = ackley
    elif problem_str == 'mishra':
        problem = mishra
    else:
        raise ValueError('Unknown problem_str %s.' % problem_str)
    return problem(**kwargs)


class BaseConstrainedSingleObjectiveProblem:
    def __init__(self, dim, **kwargs):
        self.dim = dim

    def evaluate_config(self, config, optimizer='smac'):
        raise NotImplementedError

    def evaluate(self, X: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def get_config_dict(config, optimizer='smac'):
        if optimizer == 'smac':
            config_dict = config.get_dictionary()
        elif optimizer == 'tpe':
            config_dict = config
        else:
            raise ValueError('Unknown optimizer %s' % optimizer)
        return config_dict

    @staticmethod
    def checkX(X: np.ndarray):
        X = np.atleast_2d(X)
        assert len(X.shape) == 2 and X.shape[0] == 1
        X = X.flatten()
        return X

    def get_configspace(self, optimizer='smac'):
        raise NotImplementedError


class keane(BaseConstrainedSingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(dim=10, **kwargs)
        self.lb = 0
        self.ub = 10
        self.bounds = [(self.lb, self.ub)] * self.dim
        self.num_constraints = 2

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        X = np.array([config_dict['x%s' % i] for i in range(1, 10 + 1)])
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        result = dict()
        cosX2 = np.cos(X) ** 2
        up = np.abs(np.sum(cosX2 ** 2) - 2 * np.prod(cosX2))
        down = np.sqrt(np.sum(np.arange(1, 10 + 1) * X ** 2))
        result['objs'] = [-up / down, ]
        result['constraints'] = [0.75 - np.prod(X), np.sum(X) - 7.5 * 10, ]
        return result

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            cs.add_hyperparameters(
                [UniformFloatHyperparameter("x%s" % i, self.lb, self.ub) for i in range(1, 1 + 10)])
            return cs
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = gpflowopt.domain.ContinuousParameter('x1', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x2', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x3', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x4', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x5', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x6', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x7', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x8', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x9', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x10', self.lb, self.ub)
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class ackley(BaseConstrainedSingleObjectiveProblem):
    def __init__(self, lb=-5, ub=10, **kwargs):     # -15, 30?
        super().__init__(dim=2, **kwargs)
        self.lb = lb
        self.ub = ub
        self.bounds = [(self.lb, self.ub)] * self.dim
        self.num_constraints = 1

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x1 = config_dict['x1']
        x2 = config_dict['x2']
        X = np.array([x1, x2])
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        result = dict()
        a = 20
        b = 0.2
        c = 2 * np.pi
        t1 = -a * np.exp(-b * np.sqrt(np.mean(X ** 2)))
        t2 = -np.exp(np.mean(np.cos(c * X)))
        t3 = a + np.exp(1)
        result['objs'] = [t1 + t2 + t3, ]
        result['constraints'] = [np.sign(np.sum(X)) + np.sign(np.sum(X ** 2) - 25) + 1.5, ]
        return result

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            cs.add_hyperparameters(
                [UniformFloatHyperparameter("x%s" % i, self.lb, self.ub) for i in range(1, 1 + 2)])
            return cs
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = gpflowopt.domain.ContinuousParameter('x1', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x2', self.lb, self.ub)
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class mishra(BaseConstrainedSingleObjectiveProblem):
    def __init__(self, **kwargs):
        super().__init__(dim=2, **kwargs)
        self.lb = -2 * 3.14
        self.ub = 2 * 3.14
        self.bounds = [(self.lb, self.ub)] * self.dim
        self.num_constraints = 1

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x1 = config_dict['x1']
        x2 = config_dict['x2']
        X = np.array([x1, x2])
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        result = dict()
        x, y = X[0], X[1]
        t1 = np.sin(y) * np.exp((1 - np.cos(x)) ** 2)
        t2 = np.cos(x) * np.exp((1 - np.sin(y)) ** 2)
        t3 = (x - y) ** 2
        result['objs'] = (t1 + t2 + t3,)
        result['constraints'] = ((X[0] + 5) ** 2 + (X[1] + 5) ** 2 - 25,)
        return result

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            cs.add_hyperparameters(
                [UniformFloatHyperparameter("x%s" % i, self.lb, self.ub) for i in range(1, 1 + 2)])
            return cs
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = gpflowopt.domain.ContinuousParameter('x1', self.lb, self.ub) + \
                     gpflowopt.domain.ContinuousParameter('x2', self.lb, self.ub)
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class townsend(BaseConstrainedSingleObjectiveProblem):

    def __init__(self, **kwargs):
        super().__init__(dim=2, **kwargs)
        self.bounds = [(-2.25, 2.5), (-2.5, 1.75)]
        self.num_constraints = 1

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x1 = config_dict['x1']
        x2 = config_dict['x2']
        X = np.array([x1, x2])
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        res = dict()
        res['objs'] = (-(np.cos((X[0] - 0.1) * X[1]) ** 2 + X[0] * np.sin(3 * X[0] + X[1])),)
        res['constraints'] = (
            -(-np.cos(1.5 * X[0] + np.pi) * np.cos(1.5 * X[1]) + np.sin(1.5 * X[0] + np.pi) * np.sin(1.5 * X[1])),)
        return res

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            x1 = UniformFloatHyperparameter("x1", -2.25, 2.5)
            x2 = UniformFloatHyperparameter("x2", -2.5, 1.75)
            cs.add_hyperparameters([x1, x2])
            return cs
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = gpflowopt.domain.ContinuousParameter('x1', -2.25, 2.5) + \
                     gpflowopt.domain.ContinuousParameter('x2', -2.5, 1.75)
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


