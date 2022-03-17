"""
example cmdline:

python test/reproduction/so/benchmark_so_hyperopt_math.py --problem branin --n 200 --rep 1 --start_id 0

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
mth = 'hyperopt'

problem = get_problem(problem_str)
cs = problem.get_configspace(optimizer='tpe')
time_limit_per_trial = 600


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def tpe_objective_function(config):
        y = problem.evaluate_config(config, optimizer='tpe')
        return y

    from hyperopt import tpe, fmin, Trials
    from hyperopt.utils import coarse_utcnow
    global_start_time = coarse_utcnow()

    trials = Trials()
    fmin(tpe_objective_function, cs, tpe.suggest, max_runs, trials=trials,
         rstate=np.random.RandomState(seed))

    config_list = [trial['misc']['vals'] for trial in trials.trials]
    perf_list = [trial['result']['loss'] for trial in trials.trials]
    time_list = [(trial['refresh_time'] - global_start_time).total_seconds() for trial in trials.trials]
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
