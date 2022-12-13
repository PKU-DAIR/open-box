import numpy as np
import matplotlib.pyplot as plt

from openbox.optimizer.generic_smbo import SMBO
from openbox.benchmark.objective_functions.synthetic import ZDT2

dim = 3
prob = ZDT2(dim=dim)

bo = SMBO(prob.evaluate,
          prob.config_space,
          num_objectives=prob.num_objectives,
          num_constraints=0,
          acq_type='ehvi',
          acq_optimizer_type='random_scipy',
          surrogate_type='gp',
          ref_point=prob.ref_point,
          max_runs=50,
          initial_runs=2*(dim+1),
          init_strategy='sobol',
          task_id='mo',
          random_state=1)
history = bo.run()

# plot pareto front
if history.num_objectives in [2, 3]:
    history.plot_pareto_front()
    plt.show()

# plot hypervolume
history.plot_hypervolumes(optimal_hypervolume=prob.max_hv, logy=True)
plt.show()
