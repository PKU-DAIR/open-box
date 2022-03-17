"""
example cmdline:

test time:
python test/reproduction/so/benchmark_parallel_smbo_lightgbm.py --datasets optdigits --n 100 --n_jobs 2 --batch_size 1 --rep 1 --start_id 0

run serial:
python test/reproduction/so/benchmark_parallel_smbo_lightgbm.py --datasets optdigits --runtime_limit 1200 --n_jobs 2 --batch_size 1 --rep 1 --start_id 0

run parallel:
python test/reproduction/so/benchmark_parallel_smbo_lightgbm.py --mth sync --datasets optdigits --runtime_limit 1200 --n_jobs 2 --batch_size 8 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pkl
from multiprocessing import Process, Manager

sys.path.insert(0, '.')
from test.reproduction.test_utils import timeit, seeds
from test.reproduction.test_utils import check_datasets
from so_benchmark_function import get_problem


# default_datasets = 'optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'
default_datasets = 'optdigits'

parser = argparse.ArgumentParser()
parser.add_argument('--mth', type=str, default='sync', choices=['sync', 'async', 'random'])
parser.add_argument('--datasets', type=str, default=default_datasets)
parser.add_argument('--n_jobs', type=int, default=2)
parser.add_argument('--n', type=int, default=0)
parser.add_argument('--runtime_limit', type=int, default=1200)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)

parser.add_argument('--init', type=int, default=3)
parser.add_argument('--init_strategy', type=str, default='random_explore_first')
parser.add_argument('--surrogate', type=str, default='prf', choices=['gp', 'prf'])
parser.add_argument('--optimizer', type=str, default='local', choices=['scipy', 'local'])

args = parser.parse_args()
test_datasets = args.datasets.split(',')
mth = args.mth
max_runs = args.n
runtime_limit = args.runtime_limit
n_jobs = args.n_jobs
rep = args.rep
start_id = args.start_id

batch_size = args.batch_size
ip = args.ip
port = args.port

initial_runs = args.init
init_strategy = args.init_strategy
surrogate_type = args.surrogate
if args.optimizer == 'scipy':
    acq_optimizer_type = 'random_scipy'
elif args.optimizer == 'local':
    acq_optimizer_type = 'local_random'
else:
    raise ValueError('Unknown optimizer %s' % args.optimizer)

if batch_size == 1:
    mth = 'serial'
    if max_runs > 0:    # test serial time
        runtime_limit = 999999
    else:               # benchmark serial
        max_runs = 10000
else:
    max_runs = 10000
time_limit_per_trial = 600

try:
    data_dir = '../soln-ml/data/cls_datasets/'
    check_datasets(test_datasets, data_dir)
except Exception as e:
    data_dir = '../../soln-ml/data/cls_datasets/'
    check_datasets(test_datasets, data_dir)


def evaluate_parallel(problem, mth, batch_size, seed, ip, port):
    from openbox.core.message_queue.worker import Worker
    from test.reproduction.mqsmbo_modified import mqSMBO_modified

    assert mth in ['sync', 'async', 'random']
    print(mth, batch_size, seed)
    if port == 0:
        port = 13579 + np.random.randint(1000)
    print('ip=', ip, 'port=', port)

    def objective_function(config):
        y = problem.evaluate_config(config)
        return y

    def master_run(return_list):
        if mth == 'random':
            bo = mqSMBO_modified(None, cs,
                                 initial_runs=initial_runs,     # default: 3
                                 init_strategy='random',
                                 max_runs=10000,
                                 time_limit_per_trial=time_limit_per_trial,
                                 sample_strategy='random',
                                 parallel_strategy='async', batch_size=batch_size,
                                 ip='', port=port, task_id=task_id, random_state=seed)
        else:   # sync, async
            # Caution: only prf, local_random, median_imputation now
            bo = mqSMBO_modified(None, cs,
                                 initial_runs=initial_runs,     # default: 3
                                 init_strategy=init_strategy,   # default: random_explore_first
                                 max_runs=10000,
                                 time_limit_per_trial=time_limit_per_trial,
                                 parallel_strategy=mth, batch_size=batch_size,
                                 ip='', port=port, task_id=task_id, random_state=seed)

        bo.run_with_limit(runtime_limit)
        return_list.append((bo.config_list, bo.perf_list, bo.time_list))

    def worker_run(i):
        worker = Worker(objective_function, ip, port)
        worker.run()
        print("Worker %d exit." % (i))

    manager = Manager()
    results = manager.list()  # shared list
    master = Process(target=master_run, args=(results,))
    master.start()

    time.sleep(10)  # wait for master init
    worker_pool = []
    for i in range(batch_size):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()   # wait for master to gen result
    for w in worker_pool:   # optional if repeat=1
        w.join()

    config_list, perf_list, time_list = results[0]
    return config_list, perf_list, time_list


def evaluate(problem, seed):
    def objective_function(config):
        y = problem.evaluate_config(config)
        res = dict()
        res['config'] = config
        res['objs'] = (y,)
        res['constraints'] = None
        return res

    from openbox.optimizer.generic_smbo import SMBO
    bo = SMBO(objective_function, cs,
              surrogate_type=surrogate_type,            # default: prf
              acq_optimizer_type=acq_optimizer_type,    # default: local_random
              initial_runs=initial_runs,                # default: 3
              init_strategy=init_strategy,              # default: random_explore_first
              max_runs=max_runs,
              time_limit_per_trial=time_limit_per_trial, task_id=task_id, random_state=seed)
    # bo.run()
    config_list = []
    perf_list = []
    time_list = []
    global_start_time = time.time()
    for i in range(max_runs):
        config, trial_state, _, objs = bo.iterate()
        global_time = time.time() - global_start_time
        print(seed, i, objs, config, trial_state, 'time=', global_time)
        config_list.append(config)
        perf_list.append(objs[0])
        time_list.append(global_time)
        if global_time >= runtime_limit:
            break

    return config_list, perf_list, time_list


with timeit('%s all' % (mth,)):
    for dataset in test_datasets:
        problem_str = 'lgb_%s' % (dataset,)
        problem = get_problem(problem_str, n_jobs=n_jobs, data_dir=data_dir)
        cs = problem.get_configspace(optimizer='smac')
        task_id = '%s-%d_%s' % (mth, batch_size, problem_str)

        for run_i in range(start_id, start_id + rep):
            seed = seeds[run_i]
            with timeit('%s-%d-%s-%d-%d' % (mth, batch_size, problem_str, run_i, seed)):
                # Evaluate
                if batch_size > 1:
                    config_list, perf_list, time_list = evaluate_parallel(problem, mth, batch_size, seed, ip, port)
                else:
                    config_list, perf_list, time_list = evaluate(problem, seed)

                # Save result
                print('=' * 20)
                print(seed, mth, config_list, perf_list, time_list)
                print(seed, mth, 'best perf', np.min(perf_list))

                mth_str = '%s-%d' % (mth, batch_size)
                timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                dir_path = 'logs/parallel_benchmark_%s_%d/%s/' % (problem_str, runtime_limit, mth_str)
                file = 'benchmark_%s_%04d_%s.pkl' % (mth_str, seed, timestamp)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                with open(os.path.join(dir_path, file), 'wb') as f:
                    save_item = (config_list, perf_list, time_list)
                    pkl.dump(save_item, f)
                print(dir_path, file, 'saved!', flush=True)
