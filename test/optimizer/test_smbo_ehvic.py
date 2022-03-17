import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from openbox.optimizer.generic_smbo import SMBO
from openbox.benchmark.objective_functions.synthetic import CONSTR

num_inputs = 2
prob = CONSTR()
prob.max_hv = 92.02004226679216

acq_optimizer_type = 'random_scipy'
seed = 1
initial_runs = 2 * (num_inputs + 1)
max_runs = 100

bo = SMBO(prob.evaluate, prob.config_space,
          task_id='ehvic',
          num_objs=prob.num_objs,
          num_constraints=prob.num_constraints,
          acq_type='ehvic',
          acq_optimizer_type=acq_optimizer_type,
          surrogate_type='gp',
          ref_point=prob.ref_point,
          max_runs=max_runs,
          initial_runs=initial_runs,
          init_strategy='sobol',
          random_state=seed)
bo.run()

# plot pareto front
pareto_front = np.asarray(bo.get_history().get_pareto_front())
if pareto_front.shape[-1] in (2, 3):
    if pareto_front.shape[-1] == 2:
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
    elif pareto_front.shape[-1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2])
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
    plt.title('Pareto Front')
    plt.savefig('logs/plot_pareto_front_constr.png')
    plt.show()

# plot hypervolume
hypervolume = bo.get_history().hv_data
try:
    log_hv_diff = np.log10(prob.max_hv - np.asarray(hypervolume))
    plt.plot(log_hv_diff)
except NotImplementedError:
    plt.plot(hypervolume)
plt.xlabel('Iteration')
plt.ylabel('Log Hypervolume Difference')
plt.savefig('logs/plot_hypervolume_constr.png')
plt.show()
