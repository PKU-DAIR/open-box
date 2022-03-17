"""
example cmdline:

python test/optimizer/mo/test_mo_smbo_bc.py --mth mesmo --sample_num 1 --n 100

"""
import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import Configuration
from openbox.utils.multi_objective import Hypervolume

# set problem
from mo_benchmark_function import get_setup_bc  # todo
setup = get_setup_bc()
multi_objective_func = setup['multi_objective_func']
cs = setup['cs']
run_nsgaii = setup['run_nsgaii']
problem_str = setup['problem_str']
num_inputs = setup['num_inputs']
num_objs = setup['num_objs']
referencePoint = setup['referencePoint']
real_hv = setup['real_hv']

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--rand_prob', type=float, default=0)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--mth', type=str, default='mesmo')

args = parser.parse_args()
max_runs = args.n
rand_prob = args.rand_prob
sample_num = args.sample_num
mth = args.mth
seed = 123

# Evaluate mth
X_init = np.array([
    [ 6.66666667e-01,  3.33333333e-01],
    [ 3.33333333e-01,  6.66666667e-01],
    [ 2.22222222e-01,  2.22222222e-01],
    [ 7.77777778e-01,  7.77777778e-01],
    [ 5.55555556e-01,  0             ],
    [ 0,               5.55555556e-01],
    [ 1.00000000e+00,  4.44444444e-01],
    [ 4.44444444e-01,  1.00000000e+00],
    [ 8.88888889e-01,  1.11111111e-01],
    [ 1.11111111e-01,  8.88888889e-01],
])  # use latin hypercube
X_init = [Configuration(cs, vector=X_init[i]) for i in range(X_init.shape[0])]

bo = SMBO(multi_objective_func, cs, num_objs=num_objs, max_runs=max_runs,
          # surrogate_type='gp_rbf',    # use default
          acq_type=mth,
          # initial_configurations=X_init, initial_runs=10,
          time_limit_per_trial=60, task_id='mo', random_state=seed)
bo.config_advisor.optimizer.random_chooser.prob = rand_prob     # set rand_prob, default 0
bo.config_advisor.acquisition_function.sample_num = sample_num  # set sample_num
#bo.config_advisor.acquisition_function.random_state = seed      # set random_state
bo.config_advisor.optimizer.num_mc = 1000   # MESMO optimizer only
bo.config_advisor.optimizer.num_opt = 100   # MESMO optimizer only
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
for i in range(max_runs):
    config, trial_state, objs, trial_info = bo_r.iterate()
    print(objs, config)
    hv = Hypervolume(referencePoint).compute(bo_r.get_history().get_pareto_front())
    print('hypervolume =', hv)
    hv_diff = real_hv - hv
    print('hv diff =', hv_diff)

pf_r = np.asarray(bo_r.get_history().get_pareto_front())
print('random pareto num:', pf_r.shape[0])

# Run NSGA-II to get 'real' pareto front
cheap_pareto_front = run_nsgaii()

# Plot pareto front
plt.scatter(pf[:, 0], pf[:, 1], label=mth)
plt.scatter(pf_r[:, 0], pf_r[:, 1], label='random', marker='x')
plt.scatter(cheap_pareto_front[:, 0], cheap_pareto_front[:, 1], label='NSGA-II', marker='.', alpha=0.5)

plt.title('Pareto Front')
plt.xlabel('Objective 1 - branin')
plt.ylabel('Objective 2 - Currin')
plt.legend()
plt.show()
