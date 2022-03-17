"""
example cmdline:

python test/reproduction/so/benchmark_so_cmaes_math.py --problem branin --n 200 --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

sys.path.insert(0, os.getcwd())
from test.reproduction.so.so_benchmark_function import get_problem
from test.reproduction.test_utils import timeit, seeds

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
rep = args.rep
start_id = args.start_id
mth = 'cmaes'

problem = get_problem(problem_str)
cs = problem.get_configspace(optimizer='smac')


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def objective_function(config):
        y = problem.evaluate_config(config)
        return y

    from cma import CMAEvolutionStrategy
    from openbox.utils.util_funcs import get_types
    from openbox.utils.config_space import Configuration

    types, bounds = get_types(cs)
    assert all(types == 0)

    # Check Constant Hyperparameter
    const_idx = list()
    for i, bound in enumerate(bounds):
        if np.isnan(bound[1]):
            const_idx.append(i)

    hp_num = len(bounds) - len(const_idx)
    es = CMAEvolutionStrategy(hp_num * [0], 0.99, inopts={'bounds': [0, 1], 'seed': seed})

    global_start_time = time.time()
    global_trial_counter = 0
    config_list = []
    perf_list = []
    time_list = []
    eval_num = 0
    while eval_num < max_runs:
        X = es.ask(number=es.popsize)
        _X = X.copy()
        for i in range(len(_X)):
            for index in const_idx:
                _X[i] = np.insert(_X[i], index, 0)  # np.insert returns a copy
        # _X = np.asarray(_X)
        values = []
        for xi in _X:
            # convert array to Configuration
            config = Configuration(cs, vector=xi)
            perf = objective_function(config)
            global_time = time.time() - global_start_time
            global_trial_counter += 1
            values.append(perf)
            print('=== CMAES Trial %d: %s perf=%f global_time=%f' % (global_trial_counter, config, perf, global_time))

            config_list.append(config)
            perf_list.append(perf)
            time_list.append(global_time)
        values = np.reshape(values, (-1,))
        es.tell(X, values)
        eval_num += es.popsize

    print('===== Total evaluation times=%d. Truncate to max_runs=%d.' % (eval_num, max_runs))
    config_list = config_list[:max_runs]
    perf_list = perf_list[:max_runs]
    time_list = time_list[:max_runs]
    return config_list, perf_list, time_list


with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # Evaluate
            config_list, perf_list, time_list = evaluate(mth, run_i, seed)

            # Save result
            print('=' * 20)
            print(seed, mth, config_list, perf_list, time_list)
            print(seed, mth, 'best perf', np.min(perf_list))

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/so_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (config_list, perf_list, time_list)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!', flush=True)
