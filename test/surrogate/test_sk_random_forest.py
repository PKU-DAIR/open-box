from sklearn.ensemble import RandomForestRegressor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UniformIntegerHyperparameter
import numpy as np
from openbox.utils.config_space.util import convert_configurations_to_array

import threading
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._base import _partition_estimators


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


def _collect_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        out.append(prediction)


def predictmv(rf, X):
    check_is_fitted(rf)
    # Check data
    X = rf._validate_X_predict(X)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(rf.n_estimators, rf.n_jobs)
    print('n_jobs=', n_jobs)

    # avoid storing the output of every estimator by summing them here
    if rf.n_outputs_ > 1:
        y_hat = np.zeros((X.shape[0], rf.n_outputs_), dtype=np.float64)
    else:
        print('here, rf.n_outputs_=1')
        y_hat = np.zeros((X.shape[0]), dtype=np.float64)

    # Parallel loop
    lock = threading.Lock()
    # Parallel(n_jobs=n_jobs, verbose=rf.verbose,
    #          **_joblib_parallel_args(require="sharedmem"))(
    #     delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
    #     for e in rf.estimators_)
    #
    # y_hat /= len(rf.estimators_)
    #
    # return y_hat

    all_y_preds = list()
    Parallel(n_jobs=n_jobs, verbose=rf.verbose,
             **_joblib_parallel_args(require="sharedmem"))(
        delayed(_collect_prediction)(e.predict, X, all_y_preds, lock)
        for e in rf.estimators_)
    all_y_preds = np.asarray(all_y_preds, dtype=np.float64)
    return all_y_preds


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


n_obs = 50
n_new = 5
cs = get_cs()
cs.seed(1)
configs = cs.sample_configuration(n_obs)
new_configs = cs.sample_configuration(n_new)

X = convert_configurations_to_array(configs)
Y = np.random.RandomState(47).random(size=(n_obs,))

pX = convert_configurations_to_array(new_configs)
print('shape of pX', pX.shape)

rf = RandomForestRegressor(random_state=np.random.RandomState(47), n_estimators=3)
rf.fit(X, Y)

preds = rf.predict(pX)
print(preds)

ppp = predictmv(rf, pX)
print('final predict', ppp)

m = np.mean(ppp, axis=0)
v = np.var(ppp, axis=0)

print(m, v)
print(type(m), type(v))

from joblib import effective_n_jobs
print(effective_n_jobs(None))
