from openbox.benchmark.objective_functions.synthetic import CONSTR

prob = CONSTR()
dim = 2
initial_runs = 2 * (dim + 1)

from openbox import Optimizer
opt = Optimizer(
    prob.evaluate,
    prob.config_space,
    num_objs=prob.num_objs,
    num_constraints=prob.num_constraints,
    max_runs=100,
    surrogate_type ='gp',
    acq_type='ehvic',
    acq_optimizer_type='random_scipy',
    initial_runs=initial_runs,
    init_strategy='random',
    ref_point=prob.ref_point,
    time_limit_per_trial=10,
    task_id='moc_06',
    visualization=True,
    random_state=1,
)
opt.run()

