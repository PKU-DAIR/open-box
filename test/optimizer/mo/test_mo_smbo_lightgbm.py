"""
example cmdline:

python test/optimizer/mo/test_mo_smbo_lightgbm.py --datasets spambase --mth mesmo --sample_num 1 --time_limit 20 --n 200

"""
import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from functools import partial

sys.path.insert(0, os.getcwd())
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import Configuration
from test.test_utils import check_datasets, load_data
from openbox.utils.multi_objective import Hypervolume

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--rand_prob', type=float, default=0)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--mth', type=str, default='mesmo')
parser.add_argument('--datasets', type=str)
parser.add_argument('--time_limit', type=int)   # max time for phv calculation

args = parser.parse_args()
max_runs = args.n
rand_prob = args.rand_prob
sample_num = args.sample_num
mth = args.mth
time_limit = args.time_limit
seed = 123

dataset_str = args.datasets
dataset_list = dataset_str.split(',')
data_dir = './test/data/'
check_datasets(dataset_list, data_dir)

for dataset in dataset_list:
    # set problem
    from mo_benchmark_function import get_setup_lightgbm
    setup = get_setup_lightgbm(dataset, time_limit=time_limit)  # if time_limit is None, use pre-set
    multi_objective_func = setup['multi_objective_func']
    cs = setup['cs']
    # run_nsgaii = setup['run_nsgaii']
    problem_str = setup['problem_str']
    num_inputs = setup['num_inputs']
    num_objs = setup['num_objs']
    referencePoint = setup['referencePoint']
    real_hv = setup['real_hv']
    time_limit_per_trial = 2*setup['time_limit']

    _x, _y = load_data(dataset, data_dir)
    multi_objective_func = partial(multi_objective_func, x=_x, y=_y)

    # Evaluate mth
    bo = SMBO(multi_objective_func, cs, num_objs=num_objs, max_runs=max_runs,
              # surrogate_type='gp_rbf',    # use default
              acq_type=mth,
              # initial_configurations=X_init, initial_runs=10,
              time_limit_per_trial=time_limit_per_trial, task_id='mo', random_state=seed)
    bo.config_advisor.optimizer.random_chooser.prob = rand_prob     # set rand_prob, default 0
    bo.config_advisor.acquisition_function.sample_num = sample_num  # set sample_num
    #bo.config_advisor.acquisition_function.random_state = seed      # set random_state
    bo.config_advisor.optimizer.num_mc = 10000  # MESMO optimizer only
    bo.config_advisor.optimizer.num_opt = 10    # MESMO optimizer only
    print(mth, '===== start =====')
    # bo.run()
    hv_diffs = []
    for i in range(max_runs):
        config, trial_state, objs, trial_info = bo.iterate()
        print(i, objs, config)
        hv = Hypervolume(referencePoint).compute(bo.get_history().get_pareto_front())
        print(i, 'hypervolume =', hv)
        hv_diff = real_hv - hv
        hv_diffs.append(hv_diff)
        print(i, 'hv diff =', hv_diff)

    # Print result
    pf = np.asarray(bo.get_history().get_pareto_front())
    print(mth, 'pareto num:', pf.shape[0])
    print('real hv =', real_hv)
    print('hv_diffs:', hv_diffs)

    # Evaluate the random search.
    bo_r = SMBO(multi_objective_func, cs, num_objs=num_objs, max_runs=max_runs,
                time_limit_per_trial=60, sample_strategy='random', task_id='mo_random')
    print('Random', '='*30)
    # bo_r.run()
    hv_diffs_r = []
    for i in range(max_runs):
        config, trial_state, objs, trial_info = bo_r.iterate()
        print(objs, config)
        hv = Hypervolume(referencePoint).compute(bo_r.get_history().get_pareto_front())
        print('hypervolume =', hv)
        hv_diff = real_hv - hv
        hv_diffs_r.append(hv_diff)
        print('hv diff =', hv_diff)

    pf_r = np.asarray(bo_r.get_history().get_pareto_front())
    print('random pareto num:', pf_r.shape[0])

    print('pareto num:', pf.shape[0], pf_r.shape[0], 'hv diff:', hv_diffs[-1], hv_diffs_r[-1])

    # Plot pareto front
    plt.scatter(pf[:, 0], pf[:, 1], label=mth)
    plt.scatter(pf_r[:, 0], pf_r[:, 1], label='random', marker='x')

    plt.title('Pareto Front')
    plt.xlabel('Objective 1 - accuracy score')
    plt.ylabel('Objective 2 - training time')
    plt.legend()
    plt.show()
