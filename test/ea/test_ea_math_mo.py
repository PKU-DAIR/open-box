# License: MIT
import random

import matplotlib.pyplot as plt
from openbox import space as sp, Observation
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor

from openbox.benchmark.objective_functions.synthetic import DTLZ1
from openbox.benchmark.objective_functions.synthetic import Configuration

from openbox.core.ea.saea_advisor import SAEAAdvisor
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import convert_configurations_to_array

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

function = DTLZ1(5)
space = function.config_space

# Run
if __name__ == "__main__":
    advisors = [RegularizedEAAdvisor(
        config_space = space,
        num_objectives = 2,
        task_id = 'OpenBox',
    ), SAEAAdvisor(
        config_space = space,
        num_objectives = 2,
        task_id = 'OpenBox',
        ea = RegularizedEAAdvisor,
        ref_point = (150, 150)
    )]

    res = function(space.sample_configuration())
    dim = len(res['objectives'])

    axes = None
    histories = []

    MAX_RUNS = 200
    for advisor in advisors:
        print("Now running" + str(advisor.__class__))

        for i in trange(MAX_RUNS):
            # ask
            config = advisor.get_suggestion()
            # evaluate
            ret = function(config)
            # tell
            observation = Observation(config = config, objectives = ret['objectives'])
            advisor.update_observation(observation)
            if trange == range:
                print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))

        history = advisor.get_history()
        histories.append(history.get_incumbents())

        if dim == 1:
            axes = history.plot_convergence(ax = axes)
        elif dim == 2:
            inc = history.get_incumbents()
            inc.sort(key = lambda x: x[1][0])
            plt.plot([x[1][0] for x in inc], [x[1][1] for x in inc], label = advisor.__class__.__name__)

    params = {
        'float': {'x%d' % i: (0, 1, i / 5) for i in range(1, dim + 1)}
    }
    space_bo = sp.Space()
    space_bo.add_variables([
        sp.Real(name, *para) for name, para in params['float'].items()
    ])
    opt = SMBO(
        function,
        space_bo,
        num_constraints = 0,
        num_objectives = 2,
        surrogate_type = 'gp',
        acq_optimizer_type = 'random_scipy',
        max_runs = MAX_RUNS,
        time_limit_per_trial = 10,
        task_id = 'soc',
        acq_type = 'mesmo'
    )
    history = opt.run()

    print('BO Result')
    print(history)

    if dim == 1:
        history.plot_convergence(ax = axes, yscale = 'log', name = 'BO')
    elif dim == 2:
        inc = history.get_incumbents()
        inc.sort(key = lambda x: x[1][0])
        plt.plot([x[1][0] for x in inc], [x[1][1] for x in inc], label = 'BO')

    for i, h in enumerate(histories):
        print(advisors[i].__class__)
        print(h)

    if dim <= 2:
        plt.legend()
        plt.show()

    print('--------------------OPTIMIZATION RESULTS--------------------')

    for i, h in enumerate(histories):
        print(advisors[i].__class__.__name__)
        print(h)

    print('--------------------SAEA CORRECTNESS CHECK--------------------')

    saea = advisors[1]
    gp = saea.objective_surrogates

    # print(saea.lastX)
    print('--------------------LAST GP TRAINING DATA--------------------')

    print('total {} data'.format(len(saea.lastX)))

    print('randomly print 10 of them: (X, Y, f(X))')

    rand_data = list(zip(saea.lastX,
                         saea.lastY,
                         [function(Configuration(space, vector = saea.lastX[i])) for i in range(saea.lastX.shape[0])]))
    random.shuffle(rand_data)

    for i, x in enumerate(rand_data):
        if i >= 10:
            break
        print(x)

    print('--------------------GP PREDICTION CORRECTNESS--------------------')
    print('randomly sample 10 configs, get their gp-prediction and true value: (X, gp(X), f(X))')

    for i in range(10):
        config = space.sample_configuration()

        pred = [g.predict(convert_configurations_to_array([config])) for g in gp]
        target = function(config)
        print(config.get_array(), pred, target)
        # print(convert_configurations_to_array([config]), pred, target)
