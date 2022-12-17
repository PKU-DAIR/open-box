# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Optimizer, space as sp


# objective function
def BraninCurrin(config: sp.Configuration):
    x1, x2 = config['x1'], config['x2']
    px1 = 15 * x1 - 5
    px2 = 15 * x2

    f1 = (px2 - 5.1 / (4 * np.pi ** 2) * px1 ** 2 + 5 / np.pi * px1 - 6) ** 2 \
         + 10 * (1 - 1 / (8 * np.pi)) * np.cos(px1) + 10
    f2 = (1 - np.exp(-1 / (2 * x2))) * (2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) \
         / (100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)

    result = dict()
    result['objectives'] = [f1, f2]
    return result


if __name__ == "__main__":
    # search space
    space = sp.Space()
    x1 = sp.Real("x1", 0, 1)
    x2 = sp.Real("x2", 1e-6, 1)
    space.add_variables([x1, x2])

    # provide reference point if using EHVI method
    ref_point = [18.0, 6.0]

    # run
    opt = Optimizer(
        BraninCurrin,
        space,
        num_objectives=2,
        num_constraints=0,
        max_runs=50,
        surrogate_type='gp',
        acq_type='ehvi',
        acq_optimizer_type='random_scipy',
        initial_runs=6,
        init_strategy='sobol',
        ref_point=ref_point,
        time_limit_per_trial=10,
        task_id='mo',
        random_state=1,
    )
    history = opt.run()
    print(history)

    # plot pareto front
    if history.num_objectives in [2, 3]:
        history.plot_pareto_front()  # support 2 or 3 objectives
        plt.show()

    # plot hypervolume (optimal hypervolume of BraninCurrin is approximated using NSGA-II)
    history.plot_hypervolumes(optimal_hypervolume=59.36011874867746, logy=True)
    plt.show()
