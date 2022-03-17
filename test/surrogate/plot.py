"""
example cmdline:

python test/surrogate/plot.py --mths gp,gp_mcmc,prf,tpe,lightgbm --problem branin --n 200

"""
import sys
import pickle as pkl
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

# sns.set_style(style='whitegrid')

# plt.rc('text', usetex=True)
# plt.rc('font', **{'size': 16, 'family': 'Helvetica'})

# plt.rc('font', size=16.0, family='sans-serif')
# plt.rcParams['font.sans-serif'] = "Tahoma"

# plt.rcParams['figure.figsize'] = (8.0, 4.5)
# plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
# plt.rcParams["legend.frameon"] = True
# plt.rcParams["legend.facecolor"] = 'white'
# plt.rcParams["legend.edgecolor"] = 'gray'
# plt.rcParams["legend.fontsize"] = 16
# label_size = 24
label_size = 16

sys.path.insert(0, os.getcwd())
from test.reproduction.test_utils import descending

default_mths = 'gp,gp_mcmc,prf,tpe,lightgbm'
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--problem', type=str, default='branin')
parser.add_argument('--save', type=int, default=1)

args = parser.parse_args()
max_runs = args.n
mths = args.mths.split(',')
save = bool(args.save)


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'royalblue', 'green', 'brown', 'red', 'orange', 'yellowgreen', 'black', 'yellow']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', 'd']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('gp_mcmc'):
            fill_values(name, 0)
        elif name.startswith('gp'):
            fill_values(name, 1)
        elif name.startswith('prf'):
            fill_values(name, 2)
        elif name.startswith('tpe'):
            fill_values(name, 3)
        elif name.startswith('lightgbm'):
            fill_values(name, 4)
        else:
            print(name)
            fill_values(name, 8)
    return color_dict, marker_dict


def get_mth_legend(mth):
    mth = mth.lower()
    return mth


# set problem
problem_str = args.problem
title = problem_str
log_obj = False
log_func = np.log10
std_scale = 1.0
if problem_str == 'branin':
    title = 'Branin'
    log_obj = False
    plt.ylim(0.35, 2.0)
    plt.xlim(0, 201)
    plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.14)
    std_scale = 0.5
elif problem_str == 'ackley':
    title = 'Ackley'
    log_obj = False
    plt.ylim(-0.5, 15.0)
    plt.xlim(0, 201)
    plt.subplots_adjust(top=0.97, right=0.968, left=0.10, bottom=0.14)
elif problem_str == 'beale':
    title = 'Beale'
    log_obj = False
    plt.ylim(-0.25, 5)
    plt.xlim(0, 201)
    plt.subplots_adjust(top=0.97, right=0.968, left=0.09, bottom=0.14)
elif problem_str == 'hartmann':
    title = 'Hartmann'
    log_obj = False
    plt.ylim(-3.5, -0.25)
    plt.xlim(0, 201)
    plt.subplots_adjust(top=0.97, right=0.968, left=0.12, bottom=0.14)
else:
    print('Unknown problem: %s. Use default plot settings.' % (problem_str,))
# print(f'title={title}, log_obj={log_obj}, log_func={log_func}, std_scale={std_scale}.')

color_dict, marker_dict = fetch_color_marker(mths)
# point_num = 10000
lw = 2
markersize = 6
# markevery = int(point_num / 10)
markevery = int(max_runs / 10)
alpha = 0.2

plot_list = []
legend_list = []
for mth in mths:
    result = []
    dir_path = 'logs/benchmark_surrogate/%s_%d/%s/' % (problem_str, max_runs, mth)
    for file in os.listdir(dir_path):
        if file.startswith('benchmark_%s_' % (mth,)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                save_item = pkl.load(f)
                config_list, perf_list, time_list = save_item
            if len(perf_list) != max_runs:
                print('Error len: ', file, len(perf_list))
                continue
            result.append(descending(perf_list))
    print('result rep =', len(result), mth)
    # if log_obj:
    #     result = log_func(result)  # log
    result = np.array(result)
    mean_res = np.mean(result, axis=0)
    std_res = np.std(result, axis=0)

    # plot
    x = np.arange(len(mean_res)) + 1
    p, = plt.plot(x, mean_res, lw=lw, label=get_mth_legend(mth),
                  color=color_dict[mth], marker=marker_dict[mth],
                  markersize=markersize, markevery=markevery)
    # p = plt.errorbar(x, mean_res, yerr=std_res*std_scale, fmt='', capthick=0.5, capsize=3, errorevery=max_runs//10)
    # plt.fill_between(x, mean_res - std_res * std_scale, mean_res + std_res * std_scale, alpha=alpha,
    #                  facecolor=color_dict[mth])
    plot_list.append(p)

plt.legend(ncol=2)
plt.title(title, fontsize=18)
# plt.xlabel('\\textbf{Iteration}', fontsize=label_size)
plt.xlabel('Iteration', fontsize=label_size)
if log_obj:
    plt.ylabel('Log Objective Value', fontsize=label_size)
    plt.ylabel('Log Objective Value', fontsize=label_size)
else:
    plt.ylabel('Objective Value', fontsize=label_size)
    plt.ylabel('Objective Value', fontsize=label_size)
plt.tight_layout()
if save:
    plt.savefig('surrogate_%s.png' % problem_str, dpi=300)
else:
    plt.show()
