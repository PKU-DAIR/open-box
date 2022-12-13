# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Optimizer, space as sp


# objective function
def CONSTR(config: sp.Configuration):
    x1, x2 = config['x1'], config['x2']
    
    obj1 = x1
    obj2 = (1.0 + x2) / x1

    c1 = 6.0 - 9.0 * x1 - x2
    c2 = 1.0 - 9.0 * x1 + x2

    result = dict()
    result['objectives'] = [obj1, obj2]
    result['constraints'] = [c1, c2]
    return result


if __name__ == "__main__":
    # search space
    space = sp.Space()
    x1 = sp.Real("x1", 0.1, 10.0)
    x2 = sp.Real("x2", 0.0, 5.0)
    space.add_variables([x1, x2])

    # provide reference point if using EHVI method
    ref_point = [10.0, 10.0]

    # run
    opt = Optimizer(
        CONSTR,
        space,
        num_objectives=2,
        num_constraints=2,
        max_runs=20,
        surrogate_type='gp',
        acq_type='ehvic',
        acq_optimizer_type='random_scipy',
        initial_runs=6,
        init_strategy='sobol',
        ref_point=ref_point,
        time_limit_per_trial=10,
        task_id='moc',
        random_state=1,
    )
    history = opt.run()
    print(history)

    # plot pareto front
    if history.num_objectives in [2, 3]:
        history.plot_pareto_front()  # support 2 or 3 objectives
        plt.show()

    # plot hypervolume (optimal hypervolume of CONSTR is approximated using NSGA-II)
    history.plot_hypervolumes(optimal_hypervolume=92.02004226679216, logy=True)
    plt.show()
