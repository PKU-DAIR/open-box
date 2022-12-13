import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())

from openbox.optimizer.generic_smbo import SMBO
from openbox.benchmark.objective_functions.synthetic import BraninCurrin


prob = BraninCurrin(constrained=True)
bo = SMBO(prob.evaluate, prob.config_space,
          advisor_type='mcadvisor',
          task_id='mccmo',
          num_objectives=prob.num_objectives,
          num_constraints=prob.num_constraints,
          acq_type='mcparegoc',
          ref_point=prob.ref_point,
          max_runs=100, random_state=2)
history = bo.run()

# plot pareto front
if history.num_objectives in [2, 3]:
    history.plot_pareto_front()
    plt.show()

# plot hypervolume
history.plot_hypervolumes(optimal_hypervolume=prob.max_hv, logy=True)
plt.show()
