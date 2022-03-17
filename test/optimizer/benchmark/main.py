import os
import sys
import time
import pickle
import argparse
import tabulate
import numpy as np

sys.path.append(os.getcwd())


parser = argparse.ArgumentParser()
dataset_set = 'dna,pollen,abalone,splice,madelon,spambase,wind,page-blocks(1),pc2,segment'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--algo_id', type=str, default='random_forest,adaboost')
parser.add_argument('--methods', type=str, default='openbox,smac,hyperopt')
parser.add_argument('--rep_num', type=int, default=10)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)

save_dir = './data/benchmark_results/exp1/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

args = parser.parse_args()


if __name__ == "__main__":
    for dataset in dataset_list:
        for algo_id in algo_ids:
            for run_id in range(start_id, start_id + rep):
                seed = int(seeds[run_id])
                if mth == '':
                    evaluate_hmab(algorithms, run_id, dataset=dataset, seed=seed,
                                  eval_type=eval_type,
                                  time_limit=time_limit,
                                  enable_ens=enable_ensemble)
                elif mth == 'ausk':
                    evaluate_autosklearn(algorithms, run_id,
                                         dataset=dataset, time_limit=time_limit, seed=seed,
                                         enable_ens=enable_ensemble,
                                         eval_type=eval_type)
                else:
                    raise ValueError('Invalid method name: %s.' % mth)
