import argparse
from openbox.benchmark.objective_functions.synthetic import Branin
from openbox.optimizer.generic_smbo import SMBO

parser = argparse.ArgumentParser()
parser.add_argument('--surrogate', type=str, default='gp', choices=['gp', 'gp_mcmc', 'prf', 'lightgbm', 'tpe',
                                                                    'sk_prf'])
args = parser.parse_args()

branin = Branin()
bo = SMBO(branin.evaluate,      # objective function
          branin.config_space,  # config space
          num_objs=branin.num_objs,  # number of objectives
          num_constraints=branin.num_constraints,  # number of constraints
          max_runs=50,          # number of optimization rounds
          surrogate_type=args.surrogate,
          time_limit_per_trial=180,
          task_id='quick_start')
history = bo.run()
print(history)
