import os
import sys
import numpy as np
import pandas as pd
import argparse

from functools import partial
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.getcwd())
from openbox.optimizer.message_queue_smbo import mqSMBO
from openbox.core.message_queue.worker import Worker
from test.test_utils import check_datasets, load_data


def get_cs():
    cs = ConfigurationSpace()
    n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
    max_depth = Constant('max_depth', 15)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
    subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
    cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                            colsample_bytree])
    return cs


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = LightGBM(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return 1 - balanced_accuracy_score(y_test, y_pred)


class LightGBM:
    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, colsample_bytree, random_state=None):
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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='spambase')
parser.add_argument('--n', type=int, default=50)
parser.add_argument('--role', type=str, choices=['master', 'worker'])
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=13579)
parser.add_argument('--parallel', type=str, default="sync")

args = parser.parse_args()
role = args.role
ip = args.ip
port = args.port

data_dir = './test/data/'

# check_datasets(dataset_list, data_dir)
cs = get_cs()

if role == 'master':
    run_count = args.n
    batch_size = args.batch_size
    parallel = args.parallel

    bo = mqSMBO(None, cs, max_runs=run_count, time_limit_per_trial=60, logging_dir='logs',
                parallel_strategy=parallel, batch_size=batch_size, ip='', port=port, task_id='test_mqsmbo')
    bo.run()
    inc_value = bo.get_incumbent()
    print('Message Queue SMBO', '=' * 30)
    print(inc_value)
else:
    dataset = args.dataset

    _x, _y = load_data(dataset, data_dir)
    eval = partial(eval_func, x=_x, y=_y)
    worker = Worker(eval, ip, port)
    worker.run()
