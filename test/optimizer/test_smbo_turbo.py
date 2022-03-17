import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from openbox.optimizer.generic_smbo import SMBO
from openbox.benchmark.objective_functions.synthetic import Ackley

num_inputs = 10
acq_optimizer_type = 'random_scipy'
seed = 1
prob = Ackley(dim=num_inputs, constrained=False)
initial_runs = 2 * (num_inputs + 1)
max_runs = 250

bo = SMBO(prob.evaluate, prob.config_space,
          task_id='turbo',
          advisor_type='mcadvisor',
          num_objs=prob.num_objs,
          num_constraints=prob.num_constraints,
          acq_type='mcei',
          acq_optimizer_type=acq_optimizer_type,
          use_trust_region=True,
          surrogate_type='gp',
          max_runs=max_runs,
          initial_runs=initial_runs,
          init_strategy='latin_hypercube',
          random_state=seed)
bo.run()

values = list(bo.get_history().data.values())
plt.plot(values)
plt.show()
