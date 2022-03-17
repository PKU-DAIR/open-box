import numpy as np
import matplotlib.pyplot as plt
from openbox import NSGAOptimizer, sp
from openbox.utils.multi_objective.hypervolume import Hypervolume

import sys
sys.path.insert(0, '.')
from test.reproduction.mo.mo_benchmark_function import branincurrin, plot_pf


problem = branincurrin()
space = problem.get_configspace()

# Run
opt = NSGAOptimizer(
    problem.evaluate_config, space,
    num_constraints=0,
    num_objs=2,
    max_runs=2500,
    task_id='test_nsga',
)
opt.run()

# plot
pareto_set, pareto_front = opt.get_incumbent()
plot_pf(problem, 'bc', 'nsgaii', pareto_front)

hv = Hypervolume(problem.ref_point).compute(pareto_front)
print(hv)
