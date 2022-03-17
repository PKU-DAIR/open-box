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
          num_objs=prob.num_objs,
          num_constraints=prob.num_constraints,
          acq_type='mcparegoc',
          ref_point=prob.ref_point,
          max_runs=100, random_state=2)
bo.run()

hvs = bo.get_history().hv_data
log_hv_diff = np.log10(prob.max_hv - np.asarray(hvs))
plt.plot(log_hv_diff)
plt.show()
