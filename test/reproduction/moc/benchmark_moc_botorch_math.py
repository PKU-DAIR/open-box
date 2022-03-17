"""
example cmdline:

python test/reproduction/moc/benchmark_moc_botorch_math.py --problem c2dtlz2-3-2 --refit 1 --n 200 --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

#sys.path.insert(0, '../botorch/')   # for dev

sys.path.insert(0, os.getcwd())
from moc_benchmark_function import get_problem, plot_pf
from test.reproduction.test_utils import timeit, seeds

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--init', type=int, default=0)
parser.add_argument('--mc_samples', type=int, default=128)
parser.add_argument('--refit', type=int, default=1)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--plot_mode', type=int, default=0)

args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
initial_runs = args.init
MC_SAMPLES = args.mc_samples
refit = args.refit
rep = args.rep
start_id = args.start_id
plot_mode = args.plot_mode
mth = 'botorch'

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.optim import optimize_acqf
from botorch.exceptions import BadInitialCandidatesWarning
import warnings

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

problem = get_problem(problem_str)
if initial_runs == 0:
    initial_runs = 2 * (problem.dim + 1)

# Caution: all train_x in [0, 1]. unnormalize in objective funtion and when saving
# Caution: botorch maximize the objective function
problem_bounds = torch.tensor(problem.bounds, **tkwargs).transpose(-1, -2)  # caution
standard_bounds = torch.tensor([[0.0] * problem.dim,
                                [1.0] * problem.dim], **tkwargs)
problem.ref_point = -1 * torch.tensor(problem.ref_point, **tkwargs)  # caution
hv = Hypervolume(ref_point=problem.ref_point)

INFEASIBLE_OBJ_VALUE = 9999999.0    # set obj value (to be minimized) if infeasible when saving results


# ===== botorch helper functions =====

def generate_initial_data(init_num, obj_func, time_list, global_start_time):
    # generate training data. caution: train_x in [0, 1]
    train_x = draw_sobol_samples(bounds=standard_bounds, n=1, q=init_num,
                                 seed=torch.randint(1000000, (1,)).item()).squeeze(0)
    train_obj = []
    train_con = []
    for x in train_x:
        res = obj_func(x)
        y = res['objs']
        c = res['constraints']
        train_obj.append(y)
        train_con.append(c)
        global_time = time.time() - global_start_time
        time_list.append(global_time)
    train_obj = torch.tensor(train_obj, **tkwargs).reshape(init_num, -1)
    train_con = torch.tensor(train_con, **tkwargs).reshape(init_num, -1)
    return train_x, train_obj, train_con


def initialize_model(train_x, train_obj, train_con, state_dict=None):
    # define models for objective and constraint
    train_y = torch.cat([train_obj, train_con], dim=-1)
    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(acq_func, obj_func, time_list, global_start_time):
    """Optimizes the acquisition function, and returns a new candidate and observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,  # used for initialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values
    new_x = candidates.detach()
    new_obj = []
    new_con = []
    for x in new_x:
        res = obj_func(x)
        y = res['objs']
        c = res['constraints']
        new_obj.append(y)
        new_con.append(c)
        global_time = time.time() - global_start_time
        time_list.append(global_time)
    new_obj = torch.tensor(new_obj, **tkwargs).reshape(new_x.shape[0], -1)
    new_con = torch.tensor(new_con, **tkwargs).reshape(new_x.shape[0], -1)
    print(f'evaluate {new_x.shape[0]} configs on real objective')
    return new_x, new_obj, new_con


def constraint_callable_list(num_constraints, num_objs):
    return [lambda Z: Z[..., i+num_objs] for i in range(num_constraints)]


# ===== end of botorch helper functions =====

# fix bug
def expand_initial_data(train_x, train_obj, train_con, obj_func, time_list, global_start_time):
    new_x = torch.rand(1, problem.dim, **tkwargs)   # caution: new_x in [0, 1]
    new_obj = []
    new_con = []
    for x in new_x:
        res = obj_func(x)
        y = res['objs']
        c = res['constraints']
        new_obj.append(y)
        new_con.append(c)
        global_time = time.time() - global_start_time
        time_list.append(global_time)
    new_obj = torch.tensor(new_obj, **tkwargs).reshape(new_x.shape[0], -1)
    new_con = torch.tensor(new_con, **tkwargs).reshape(new_x.shape[0], -1)

    # update training points
    train_x = torch.cat([train_x, new_x])
    train_obj = torch.cat([train_obj, new_obj])
    train_con = torch.cat([train_con, new_con])
    return train_x, train_obj, train_con


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def objective_function(x: torch.Tensor):
        # Caution: unnormalize and maximize
        x = unnormalize(x, bounds=problem_bounds)
        x = x.cpu().numpy().astype(np.float64)      # caution
        res = problem.evaluate(x)
        res['objs'] = [-y for y in res['objs']]
        return res  # Caution: negative values imply feasibility in botorch

    hv_diffs = []
    time_list = []
    global_start_time = time.time()

    # random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # call helper functions to generate initial training data and initialize model
    train_x, train_obj, train_con = generate_initial_data(initial_runs, objective_function, time_list, global_start_time)
    # fix bug: find feasible
    real_initial_runs = initial_runs
    while real_initial_runs < max_runs:
        # compute feasible observations
        is_feas = (train_con <= 0).all(dim=-1)
        # compute points that are better than the known reference point
        better_than_ref = (train_obj > problem.ref_point).all(dim=-1)
        if (is_feas & better_than_ref).any():
            break
        train_x, train_obj, train_con = expand_initial_data(train_x, train_obj, train_con,
                                                            objective_function, time_list, global_start_time)
        real_initial_runs += 1
        print('=== Expand initial data to find feasible. Iter =', real_initial_runs)
    mll, model = initialize_model(train_x, train_obj, train_con)

    # for plot
    X_init = train_x.cpu().numpy().astype(np.float64)
    Y_init = -1 * train_obj.cpu().numpy().astype(np.float64)
    # calculate hypervolume of init data
    for i in range(real_initial_runs):
        train_obj_i = train_obj[:i+1]
        train_con_i = train_con[:i+1]
        # compute pareto front
        is_feas_i = (train_con_i <= 0).all(dim=-1)
        feas_train_obj_i = train_obj_i[is_feas_i]
        pareto_mask = is_non_dominated(feas_train_obj_i)
        pareto_y = feas_train_obj_i[pareto_mask]
        # compute hypervolume
        volume = hv.compute(pareto_y)
        hv_diff = problem.max_hv - volume
        hv_diffs.append(hv_diff)

    # run (max_runs - real_initial_runs) rounds of BayesOpt after the initial random batch
    for iteration in range(real_initial_runs + 1, max_runs + 1):
        t0 = time.time()
        try:
            # fit the models
            fit_gpytorch_model(mll)

            # define the qEHVI acquisition modules using a QMC sampler
            sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            # compute feasible observations
            is_feas = (train_con <= 0).all(dim=-1)
            # compute points that are better than the known reference point
            better_than_ref = (train_obj > problem.ref_point).all(dim=-1)
            # partition non-dominated space into disjoint rectangles
            partitioning = NondominatedPartitioning(
                num_outcomes=problem.num_objs,
                # use observations that are better than the specified reference point and feasible
                Y=train_obj[better_than_ref & is_feas],
            )
            qEHVI = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=problem.ref_point.tolist(),  # use known reference point
                partitioning=partitioning,
                sampler=sampler,
                # define an objective that specifies which outcomes are the objectives
                objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.num_objs))),
                # specify that the constraint is on the last outcome
                constraints=constraint_callable_list(problem.num_constraints, num_objs=problem.num_objs),
            )
            # optimize and get new observation
            new_x, new_obj, new_con = optimize_acqf_and_get_observation(qEHVI, objective_function, time_list, global_start_time)
        except Exception as e:  # handle numeric problem
            step = 2
            print('===== Exception in optimization loop, restart with 1/%d of training data: %s' % (step, str(e)))
            if refit == 1:
                mll, model = initialize_model(train_x[::step], train_obj[::step], train_con[::step])
            else:
                mll, model = initialize_model(
                    train_x[::step],
                    train_obj[::step],
                    train_con[::step],
                    model.state_dict(),
                )
            # fit the models
            fit_gpytorch_model(mll)

            # define the qEHVI acquisition modules using a QMC sampler
            sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            # compute feasible observations
            is_feas = (train_con[::step] <= 0).all(dim=-1)
            # compute points that are better than the known reference point
            better_than_ref = (train_obj[::step] > problem.ref_point).all(dim=-1)
            # partition non-dominated space into disjoint rectangles
            partitioning = NondominatedPartitioning(
                num_outcomes=problem.num_objs,
                # use observations that are better than the specified reference point and feasible
                Y=train_obj[::step][better_than_ref & is_feas],
            )
            qEHVI = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=problem.ref_point.tolist(),  # use known reference point
                partitioning=partitioning,
                sampler=sampler,
                # define an objective that specifies which outcomes are the objectives
                objective=IdentityMCMultiOutputObjective(outcomes=list(range(problem.num_objs))),
                # specify that the constraint is on the last outcome
                constraints=constraint_callable_list(problem.num_constraints, num_objs=problem.num_objs),
            )
            # optimize and get new observation
            new_x, new_obj, new_con = optimize_acqf_and_get_observation(qEHVI, objective_function, time_list,
                                                                        global_start_time)
            assert len(time_list) == iteration

        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_con = torch.cat([train_con, new_con])

        # update progress
        # compute pareto front
        is_feas = (train_con <= 0).all(dim=-1)
        feas_train_obj = train_obj[is_feas]
        pareto_mask = is_non_dominated(feas_train_obj)
        pareto_y = feas_train_obj[pareto_mask]
        # compute hypervolume
        volume = hv.compute(pareto_y)
        hv_diff = problem.max_hv - volume
        hv_diffs.append(hv_diff)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        # Note: they find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        if refit == 1:
            mll, model = initialize_model(train_x, train_obj, train_con)
        else:
            mll, model = initialize_model(
                train_x,
                train_obj,
                train_con,
                model.state_dict(),
            )

        t1 = time.time()
        print("Iter %d: x=%s, perf=%s, con=%s, hv_diff=%f, time=%.2f, global_time=%.2f"
              % (iteration, unnormalize(new_x, bounds=problem_bounds), -new_obj, new_con, hv_diff,
                 t1-t0, time_list[-1]), flush=True)

    # compute pareto front
    is_feas = (train_con <= 0).all(dim=-1)
    feas_train_obj = train_obj[is_feas]
    pareto_mask = is_non_dominated(feas_train_obj)
    pareto_y = feas_train_obj[pareto_mask]
    pf = -1 * pareto_y.cpu().numpy().astype(np.float64)
    # Save result
    X = unnormalize(train_x, bounds=problem_bounds).cpu().numpy().astype(np.float64)  # caution
    train_obj[~is_feas] = -INFEASIBLE_OBJ_VALUE  # set infeasible
    Y = -1 * train_obj.cpu().numpy().astype(np.float64)

    # plot for debugging
    if plot_mode == 1:
        plot_pf(problem, problem_str, mth, pf, Y_init)

    return hv_diffs, pf, X, Y, time_list


with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # Evaluate
            hv_diffs, pf, X, Y, time_list = evaluate(mth, run_i, seed)

            # Save result
            print('=' * 20)
            print(seed, mth, X, Y, time_list, hv_diffs)
            print(seed, mth, 'best hv_diff:', hv_diffs[-1])
            print(seed, mth, 'max_hv:', problem.max_hv)
            if pf is not None:
                print(seed, mth, 'pareto num:', pf.shape[0])

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/moc_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (hv_diffs, pf, X, Y, time_list)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!', flush=True)
