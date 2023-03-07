import time
from functools import partial
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter
from sklearn.decomposition import KernelPCA
from openbox.utils.limit import run_obj_func


def objective_function(config: Configuration, sleep):
    print(f'obj get config: {config}. sleep: {sleep}')
    # Caution: don't use dill (macOS python>=3.8), or you need to re-import numpy in obj_func
    print('test numpy in multiprocess:', np.array([1]))
    if sleep:
        time.sleep(5)
    result = dict(objs=[1, 2], constraints=[3, 4])
    return result


def error_obj_func(*args, **kwargs):
    raise ValueError('error obj func')


def kpca_obj_func(*args, **kwargs):
    # change mat_n: {10000, 1000, 100, 10}
    mat_n = 1000
    m = np.random.random((mat_n, mat_n))

    for _ in range(1000):
        pca = KernelPCA()
        pca.fit_transform(m)
    return m * m


def test_time_limit():
    cs = ConfigurationSpace()
    hp1 = CategoricalHyperparameter('hp1', choices=['a', 'b', 'c'])
    hp2 = UniformFloatHyperparameter('hp2', lower=0, upper=100)
    cs.add_hyperparameters([hp1, hp2])
    config = cs.sample_configuration()

    start = time.time()
    time_limit = 3.5   # max: 2147483 ?
    print('\n=====', 'run no sleep')
    obj_args, obj_kwargs = (config, ), dict(sleep=False)
    result = run_obj_func(objective_function, obj_args, obj_kwargs, time_limit)
    print(result, 'time:', time.time() - start)

    print('\n=====', 'run sleep')
    obj_args, obj_kwargs = (config, ), dict(sleep=True)
    result = run_obj_func(objective_function, obj_args, obj_kwargs, time_limit)
    print(result, 'time:', time.time() - start)

    print('\n=====', 'run error func')
    obj_args, obj_kwargs = (config,), dict(sleep=True)
    result = run_obj_func(error_obj_func, obj_args, obj_kwargs, time_limit)
    print(result, 'time:', time.time() - start)
    print(result['traceback'])

    print('\n=====', 'run no time limit')
    obj_args, obj_kwargs = (config,), dict(sleep=True)
    result = run_obj_func(objective_function, obj_args, obj_kwargs)
    print(result, 'time:', time.time() - start)

    print('\n=====', 'run timeout=0')
    obj_args, obj_kwargs = (config,), dict(sleep=True)
    result = run_obj_func(objective_function, obj_args, obj_kwargs, timeout=0)
    print(result, 'time:', time.time() - start)

    print('\n=====', 'run partial func')
    obj_args, obj_kwargs = (config,), dict()
    obj2 = partial(objective_function, sleep=True)
    result = run_obj_func(obj2, obj_args, obj_kwargs, timeout=3)
    print(result, 'time:', time.time() - start)

    print('\n=====', 'run kpca, timeout=2.5')
    obj_args, obj_kwargs = (), dict()
    result = run_obj_func(kpca_obj_func, obj_args, obj_kwargs, timeout=2.5)
    print(result, 'time:', time.time() - start)
