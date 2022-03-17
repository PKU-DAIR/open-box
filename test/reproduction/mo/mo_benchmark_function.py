import numpy as np

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter, InCondition, EqualsCondition, UnParametrizedHyperparameter, \
    ForbiddenEqualsClause, ForbiddenInClause, ForbiddenAndConjunction

from openbox.benchmark.objective_functions.synthetic import DTLZ1, DTLZ2, BraninCurrin, VehicleSafety, ZDT1, ZDT2, ZDT3


def get_problem(problem_str, **kwargs):
    problem = None
    if problem_str.startswith('dtlz1'):
        params = problem_str.split('-')
        assert params[0] == 'dtlz1'
        if len(params) == 1:
            return dtlz1(dim=5, num_objs=4)
        elif len(params) == 3:
            return dtlz1(dim=int(params[1]), num_objs=int(params[2]))
    elif problem_str.startswith('dtlz2'):
        params = problem_str.split('-')
        assert params[0] == 'dtlz2'
        if len(params) == 1:
            return dtlz2(dim=12, num_objs=2)
        elif len(params) == 3:
            return dtlz2(dim=int(params[1]), num_objs=int(params[2]))
    elif problem_str == 'branincurrin':
        problem = branincurrin
    elif problem_str == 'vehiclesafety':
        problem = vehiclesafety
    elif problem_str.startswith('zdt'):
        params = problem_str.split('-')
        assert params[0] in ('zdt1', 'zdt2', 'zdt3')
        if len(params) == 1:
            return zdt(problem_str=params[0], dim=3)
        else:
            return zdt(problem_str=params[0], dim=int(params[1]))
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


class BaseMultiObjectiveProblem:
    def __init__(self, dim, num_objs, problem=None, **kwargs):
        self.dim = dim
        self.num_objs = num_objs
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


class dtlz1(BaseMultiObjectiveProblem):

    def __init__(self, dim, num_objs, **kwargs):
        problem = DTLZ1(dim=dim, num_objs=num_objs)
        super().__init__(dim=dim, num_objs=num_objs, problem=problem, **kwargs)
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
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = np.sum([
                gpflowopt.domain.ContinuousParameter('x%d' % i, self.lb, self.ub) for i in range(1, self.dim+1)
            ])
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class dtlz2(BaseMultiObjectiveProblem):

    def __init__(self, dim, num_objs, **kwargs):
        problem = DTLZ2(dim=dim, num_objs=num_objs)
        super().__init__(dim=dim, num_objs=num_objs, problem=problem, **kwargs)
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
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = np.sum([
                gpflowopt.domain.ContinuousParameter('x%d' % i, self.lb, self.ub) for i in range(1, self.dim+1)
            ])
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class branincurrin(BaseMultiObjectiveProblem):

    def __init__(self, **kwargs):
        problem = BraninCurrin()
        super().__init__(dim=2, num_objs=2, problem=problem, **kwargs)
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
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = np.sum([
                gpflowopt.domain.ContinuousParameter('x%d' % i, self.lb, self.ub) for i in range(1, self.dim+1)
            ])
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class vehiclesafety(BaseMultiObjectiveProblem):

    def __init__(self, **kwargs):
        problem = VehicleSafety()
        super().__init__(dim=5, num_objs=3, problem=problem, **kwargs)
        self.lb = 1
        self.ub = 3
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
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = np.sum([
                gpflowopt.domain.ContinuousParameter('x%d' % i, self.lb, self.ub) for i in range(1, self.dim+1)
            ])
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)


class zdt(BaseMultiObjectiveProblem):

    def __init__(self, problem_str, dim, **kwargs):
        if problem_str == 'zdt1':
            problem = ZDT1
        elif problem_str == 'zdt2':
            problem = ZDT2
        elif problem_str == 'zdt3':
            problem = ZDT3
        else:
            raise ValueError
        problem = problem(dim=dim)
        super().__init__(dim=dim, num_objs=2, problem=problem, **kwargs)
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
        elif optimizer == 'gpflowopt':
            import gpflowopt
            domain = np.sum([
                gpflowopt.domain.ContinuousParameter('x%d' % i, self.lb, self.ub) for i in range(1, self.dim+1)
            ])
            return domain
        else:
            raise ValueError('Unknown optimizer %s when getting configspace' % optimizer)

