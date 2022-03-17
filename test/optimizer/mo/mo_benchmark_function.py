import numpy as np
import math
from copy import deepcopy
import time

from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from platypus import NSGAII, Problem, Real


# === benchmark 1: bc
def branin(x1):
    x = deepcopy(x1)
    x = np.asarray(x)
    assert x.shape == (2,)
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    return float(
        np.square(x[1] - (5.1 / (4 * np.square(math.pi))) * np.square(x[0]) + (5 / math.pi) * x[0] - 6) + 10 * (
                1 - (1. / (8 * math.pi))) * np.cos(x[0]) + 10)


def Currin(x):
    x = np.asarray(x)
    assert x.shape == (2,)
    try:
        return float(((1 - math.exp(-0.5 * (1 / x[1]))) * (
                (2300 * pow(x[0], 3) + 1900 * x[0] * x[0] + 2092 * x[0] + 60) / (
                100 * pow(x[0], 3) + 500 * x[0] * x[0] + 4 * x[0] + 20))))
    except Exception as e:
        return 300


def get_setup_bc():
    # test scale
    # scale1 = [-100, 100]
    # scale2 = [0, 10]
    scale1 = [0, 1]
    scale2 = [0, 1]

    def multi_objective_func_bc(config):
        xs = config.get_dictionary()
        x0 = (xs['x0'] + scale1[0]) / (scale1[0] + scale1[1])
        x1 = (xs['x1'] + scale2[0]) / (scale2[0] + scale2[1])
        x = [x0, x1]  # x0, x1 in [0, 1]. test scale
        y1 = branin(x)
        y2 = Currin(x)
        res = dict()
        res['config'] = config
        res['objs'] = (y1, y2)
        res['constraints'] = None
        return res

    def get_cs_bc():
        cs_bc = ConfigurationSpace()
        x0 = UniformFloatHyperparameter("x0", scale1[0], scale1[1])
        # x0 = UniformIntegerHyperparameter("x0", scale1[0], scale1[1])  # test int
        x1 = UniformFloatHyperparameter("x1", scale2[0], scale2[1])
        cs_bc.add_hyperparameters([x0, x1])
        return cs_bc

    def run_nsgaii_bc():
        def CMO(xi):
            xi = np.asarray(xi)
            y = [branin(xi), Currin(xi)]
            return y

        problem = Problem(2, 2)
        problem.types[:] = Real(0, 1)
        problem.function = CMO
        algorithm = NSGAII(problem)
        algorithm.run(2500)
        cheap_pareto_front = np.array([list(solution.objectives) for solution in algorithm.result])
        return cheap_pareto_front

    problem_str = 'bc'
    num_inputs = 2
    num_objs = 2
    referencePoint = [1e5] * num_objs   # must greater than max value of objective
    real_hv = 1e10

    setup = dict(
        multi_objective_func=multi_objective_func_bc,
        cs=get_cs_bc(),
        run_nsgaii=run_nsgaii_bc,
        problem_str=problem_str,
        num_inputs=num_inputs,
        num_objs=num_objs,
        referencePoint=referencePoint,
        real_hv=real_hv,
    )
    return setup


# === benchmark 2 todo


# === benchmark 3 lightgbm hyper-parameter tuning
class LightGBM:
    def __init__(self, n_estimators, learning_rate, num_leaves, min_child_samples,
                 subsample, colsample_bytree, max_depth=15, random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree

        self.n_jobs = 2
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        min_child_samples=self.min_child_samples,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)


def get_setup_lightgbm(dataset, time_limit=None):
    # classification only
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

    time_limit_dict = dict(
        spambase=20,
    )

    def multi_objective_func_lightgbm(config, x, y):
        """
        Caution:
            from functools import partial
            multi_objective_func = partial(multi_objective_func, x=x, y=y)
        """
        start_time = time.time()

        params = config.get_dictionary()
        model = LightGBM(**params)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        time_taken = time.time() - start_time
        #acc = -accuracy_score(y_test, y_pred)  # minimize
        #f1 = -f1_score(y_test, y_pred)  # minimize todo average
        acc = -balanced_accuracy_score(y_test, y_pred)  # minimize
        #f1 = -f1_score(y_test, y_pred, average='macro')  # minimize todo average

        y1 = acc            # remember to change referencePoint and real_hv
        y2 = time_taken

        res = dict()
        res['config'] = config
        res['objs'] = (y1, y2)
        res['constraints'] = None

        if any(res['objs'][i] > referencePoint[i] for i in range(len(referencePoint))):
            print('[ERROR]=== objective evaluate error! objs =', res['objs'], 'referencePoint =', referencePoint)
            res['objs'] = [ref - 1e-5 for ref in referencePoint]
        return res

    def get_cs_lightgbm():  # todo q and int for compare?
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
        num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
        # max_depth = Constant('max_depth', 15)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
        min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
        subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
        # cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
        #                         colsample_bytree])
        cs.add_hyperparameters([n_estimators, num_leaves, learning_rate, min_child_samples, subsample,
                                colsample_bytree])
        return cs

    problem_str = 'lightgbm-' + dataset
    num_inputs = 6
    num_objs = 2
    time_limit = time_limit_dict.get(dataset, time_limit)
    assert time_limit is not None, 'time_limit is not set for dataset: %s' % (dataset,)
    referencePoint = [0.5, time_limit]  # must greater than max value of objective
    real_hv = 1.5 * time_limit  # todo

    setup = dict(
        multi_objective_func=multi_objective_func_lightgbm,
        cs=get_cs_lightgbm(),
        run_nsgaii=None,
        problem_str=problem_str,
        num_inputs=num_inputs,
        num_objs=num_objs,
        referencePoint=referencePoint,
        real_hv=real_hv,
        time_limit=time_limit
    )
    return setup


# === benchmark 4

