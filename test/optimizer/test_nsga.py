from openbox import NSGAOptimizer

from openbox import space as sp

space = sp.Space()
x1 = sp.Real('x1', 0, 100)
x2 = sp.Int('x2', -100, 0)
x3 = sp.Categorical('x3', ['a', 'b', 'c'])
x4 = sp.Ordinal('x4', ['10', '20', '30', '40', '50', '60'])
space.add_variables([x1, x2, x3, x4])

def objective_func(config):
    x1, x2, x3, x4 = config['x1'], config['x2'], config['x3'], config['x4']

    x3 = 1
    x4 = int(x4)

    y1 = (x1 + x2 + x3 + 10) ** 2
    y2 = (x1 - x2 - x4 - 10) ** 2
    return dict(objs=[y1, y2])

opt = NSGAOptimizer(
    objective_func, space,
    num_constraints=0,
    num_objs=2,
    max_runs=2500,
    task_id='test_nsga',
)
opt.run()

print(opt.get_incumbent())
