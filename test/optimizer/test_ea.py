import numpy as np
from openbox import sp, Optimizer, ParallelOptimizer


# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objs': (y,)}


# Run
if __name__ == "__main__":
    max_runs = 40
    opt = Optimizer(
        branin,
        space,
        advisor_type='ea',
        max_runs=max_runs,
        time_limit_per_trial=30,
        task_id='test_ea',
    )
    history = opt.run()
    print(history)

    batch_size = 4
    opt = ParallelOptimizer(
        branin,
        space,
        parallel_strategy='async',
        batch_size=batch_size,
        sample_strategy='ea',
        max_runs=max_runs,
        time_limit_per_trial=30,
        task_id='test_ea',
    )
    history = opt.run()
    print(history)

    opt = ParallelOptimizer(
        branin,
        space,
        parallel_strategy='sync',
        batch_size=batch_size,
        sample_strategy='ea',
        max_runs=max_runs,
        time_limit_per_trial=30,
        task_id='test_ea',
    )
    history = opt.run()
    print(history)

