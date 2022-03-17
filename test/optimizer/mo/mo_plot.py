"""
example cmdline:

python test/optimizer/mo/mo_plot.py --mths mesmo-1,usemo-1,gpflowopt-hvpoi --problem bc --n 110

"""
import os
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# default_mths = 'mesmo-1,usemo-1,gpflowopt-hvpoi'
default_mths = 'usemo-1,gpflowopt-hvpoi'
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--problem', type=str, default='bc')

args = parser.parse_args()
max_runs = args.n
mths = args.mths.split(',')  # list of 'mth-sample_num'


def get_mth_legend(mth):
    mth = mth.lower()
    if mth.startswith('mesmo'):
        return 'MESMO'
    elif mth.startswith('usemo'):
        return 'USeMO'
    elif mth == 'gpflowopt-hvpoi':
        return 'GPflowOpt-HVPoI'
    else:
        return mth


# set problem
problem_str = args.problem
title = problem_str
log_phv = True
log_func = np.log10
std_scale = 1.0
if problem_str == 'bc':
    title = 'Branin-Currin'
    log_phv = True
    plt.ylim(11.9, 12.45)
    plt.xlim(0, 111)
    std_scale = 0.7
elif problem_str.startswith('lightgbm'):
    # title = problem_str
    title = 'LightGBM-Spambase'     # todo
    log_phv = False
    plt.ylim(1.1, 1.8)
    plt.xlim(0, 150)
    std_scale = 0.4
else:
    print('Unknown problem: %s. Use default plot settings.' % (problem_str,))
print(f'title={title}, log_phv={log_phv}, log_func={log_func}, std_scale={std_scale}.')

plot_list = []
legend_list = []
for mth in mths:
    result = []
    dir_path = 'logs/mo_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
    for file in os.listdir(dir_path):
        if file.startswith('benchmark_%s_' % (mth,)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                save_item = pkl.load(f)
                hv_diffs, pf, data = save_item
            if len(hv_diffs) != max_runs:
                print('Error len: ', file, len(hv_diffs))
                continue
            result.append(hv_diffs)
            print('last hv_diff =', hv_diffs[-1])
    print('result rep =', len(result), mth)
    if log_phv:
        result = log_func(result)  # log
    mean_res = np.mean(result, axis=0)
    std_res = np.std(result, axis=0)

    # plot
    x = np.arange(len(mean_res)) + 1
    p, = plt.plot(x, mean_res)
    # p = plt.errorbar(x, mean_res, yerr=std_res*std_scale, fmt='', capthick=0.5, capsize=3, errorevery=max_runs//10)
    plt.fill_between(x, mean_res - std_res*std_scale, mean_res + std_res*std_scale, alpha=0.2)
    plot_list.append(p)
    legend_list.append(get_mth_legend(mth))
    print('last mean,std:', mean_res[-1], std_res[-1])

plt.legend(plot_list, legend_list, loc='upper right', fontsize=12)
plt.title(title, fontsize=18)
plt.xlabel('Iteration', fontsize=15)
if log_phv:
    plt.ylabel('Log Hypervolume Difference', fontsize=15)
else:
    plt.ylabel('Hypervolume Difference', fontsize=15)
plt.show()
