import numpy as np

# from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, InCondition, EqualsCondition, UnParametrizedHyperparameter, \
    ForbiddenEqualsClause, ForbiddenInClause, ForbiddenAndConjunction

from openbox.benchmark.objective_functions.synthetic import DTLZ2, BraninCurrin, BNH, SRN, CONSTR


def get_problem(problem_str, **kwargs):
    problem = None
    if problem_str.startswith('c2dtlz2'):
        params = problem_str.split('-')
        assert params[0] == 'c2dtlz2'
        if len(params) == 1:
            return c2dtlz2(dim=3, num_objs=2)
        elif len(params) == 3:
            return c2dtlz2(dim=int(params[1]), num_objs=int(params[2]))
    elif problem_str == 'cbranincurrin':
        problem = cbranincurrin
    elif problem_str == 'bnh':
        problem = bnh
    elif problem_str == 'srn':
        problem = srn
    elif problem_str == 'constr':
        problem = constr
    if problem is None:
        raise ValueError('Unknown problem_str %s.' % problem_str)
    return problem(**kwargs)


def plot_pf(problem, problem_str, mth, pf, Y_init=None):
    import matplotlib.pyplot as plt
    assert problem.num_objs in (2, 3)
    if problem.num_objs == 2:
        plt.scatter(pf[:, 0], pf[:, 1], label=mth)
        if Y_init is not None:
            plt.scatter(Y_init[:, 0], Y_init[:, 1], label='init', marker='x')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
    elif problem.num_objs == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(pf[:, 0], pf[:, 1], pf[:, 2], label=mth)
        if Y_init is not None:
            ax.scatter3D(Y_init[:, 0], Y_init[:, 1], Y_init[:, 3], label='init', marker='x')
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
    else:
        raise ValueError('Cannot plot_pf with problem.num_objs == %d.' % (problem.num_objs,))
    plt.title('Pareto Front of %s' % (problem_str,))
    plt.legend()
    plt.show()


class BaseConstrainedMultiObjectiveProblem:
    def __init__(self, dim, num_objs, num_constraints, problem=None, **kwargs):
        self.dim = dim
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        if problem is not None:
            self.problem = problem
            self.ref_point = problem.ref_point
            try:
                self.max_hv = problem.max_hv
            except NotImplementedError:
                self.max_hv = 0.0

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


class c2dtlz2(BaseConstrainedMultiObjectiveProblem):

    def __init__(self, dim, num_objs, **kwargs):
        problem = DTLZ2(dim=dim, num_objs=num_objs, constrained=True)
        super().__init__(dim=dim, num_objs=num_objs, num_constraints=1, problem=problem, **kwargs)
        self.lb = 0
        self.ub = 1
        self.bounds = [(self.lb, self.ub)] * self.dim

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x_list = [config_dict['x%d' % i] for i in range(1, self.dim+1)]
        X = np.array(x_list)
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        return self.problem._evaluate(X)  # dict

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            return self.problem.config_space
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class cbranincurrin(BaseConstrainedMultiObjectiveProblem):

    def __init__(self, **kwargs):
        problem = BraninCurrin(constrained=True)
        super().__init__(dim=2, num_objs=2, num_constraints=1, problem=problem, **kwargs)
        self.lb = 1e-10  # fix numeric problem
        self.ub = 1
        self.bounds = [(self.lb, self.ub)] * self.dim

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x_list = [config_dict['x%d' % i] for i in range(1, self.dim+1)]
        X = np.array(x_list)
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        return self.problem._evaluate(X)  # dict

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            cs.add_hyperparameters(
                [UniformFloatHyperparameter("x%s" % i, self.lb, self.ub) for i in range(1, self.dim+1)])
            return cs
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class bnh(BaseConstrainedMultiObjectiveProblem):

    def __init__(self, **kwargs):
        problem = BNH()
        super().__init__(dim=2, num_objs=2, num_constraints=2, problem=problem, **kwargs)
        self.bounds = [(0.0, 5.0), (0.0, 3.0)]
        self.new_max_hv = 7242.068539049498     # this is approximated using NSGA-II

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x_list = [config_dict['x%d' % i] for i in range(1, self.dim+1)]
        X = np.array(x_list)
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        return self.problem._evaluate(X)  # dict

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            return self.problem.config_space
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class srn(BaseConstrainedMultiObjectiveProblem):

    def __init__(self, **kwargs):
        problem = SRN()
        super().__init__(dim=2, num_objs=2, num_constraints=2, problem=problem, **kwargs)
        self.lb = -20.0
        self.ub = 20.0
        self.bounds = [(self.lb, self.ub)] * self.dim
        self.new_max_hv = 34229.434882104855    # this is approximated using NSGA-II

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x_list = [config_dict['x%d' % i] for i in range(1, self.dim+1)]
        X = np.array(x_list)
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        return self.problem._evaluate(X)  # dict

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            return self.problem.config_space
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class constr(BaseConstrainedMultiObjectiveProblem):

    def __init__(self, **kwargs):
        problem = CONSTR()
        super().__init__(dim=2, num_objs=2, num_constraints=2, problem=problem, **kwargs)
        self.bounds = [(0.1, 10.0), (0.0, 5.0)]
        self.new_max_hv = 92.02004226679216     # this is approximated using NSGA-II

    def evaluate_config(self, config, optimizer='smac'):
        config_dict = self.get_config_dict(config, optimizer)
        x_list = [config_dict['x%d' % i] for i in range(1, self.dim+1)]
        X = np.array(x_list)
        return self.evaluate(X)

    def evaluate(self, X: np.ndarray):
        X = self.checkX(X)
        return self.problem._evaluate(X)  # dict

    def get_configspace(self, optimizer='smac'):
        if optimizer == 'smac':
            return self.problem.config_space
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)

