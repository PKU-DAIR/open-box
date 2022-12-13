import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from openbox.optimizer.generic_smbo import SMBO
from openbox.benchmark.objective_functions.synthetic import VehicleSafety

num_inputs = 5
num_objectives = 3
acq_optimizer_type = 'random_scipy'
seed = 1
prob = VehicleSafety()
initial_runs = 2 * (num_inputs + 1)
max_runs = 100 + initial_runs

bo = SMBO(prob.evaluate, prob.config_space,
          task_id='ehvi',
          num_objectives=prob.num_objectives,
          num_constraints=prob.num_constraints,
          acq_type='ehvi',
          acq_optimizer_type=acq_optimizer_type,
          surrogate_type='gp',
          ref_point=prob.ref_point,
          max_runs=max_runs,
          initial_runs=initial_runs,
          init_strategy='sobol',
          random_state=seed)
history = bo.run()

# plot pareto front
if history.num_objectives in [2, 3]:
    history.plot_pareto_front()
    plt.show()

# plot hypervolume
history.plot_hypervolumes(optimal_hypervolume=prob.max_hv, logy=True)
plt.show()
