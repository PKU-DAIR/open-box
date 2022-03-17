from openbox.benchmark.objective_functions.synthetic import Branin
from openbox.optimizer.generic_smbo import SMBO

branin = Branin()
bo = SMBO(branin.evaluate,      # objective function
          branin.config_space,  # config space
          num_objs=branin.num_objs,  # number of objectives
          num_constraints=branin.num_constraints,  # number of constraints
          max_runs=50,          # number of optimization rounds
          surrogate_type='gp',
          time_limit_per_trial=180,
          # acq_optimizer_type='scipy_global',
          task_id='quick_start')
history = bo.run()
print(history)
