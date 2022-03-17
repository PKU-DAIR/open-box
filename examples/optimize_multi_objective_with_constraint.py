# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Optimizer, sp


# objective function
def CONSTR(config: sp.Configuration):
    x1, x2 = config['x1'], config['x2']
    
    obj1 = x1
    obj2 = (1.0 + x2) / x1

    c1 = 6.0 - 9.0 * x1 - x2
    c2 = 1.0 - 9.0 * x1 + x2

    result = dict()
    result['objs'] = [obj1, obj2]
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
        num_objs=2,
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
    opt.run()

    # plot pareto front
    pareto_front = np.asarray(opt.get_history().get_pareto_front())
    if pareto_front.shape[-1] in (2, 3):
        if pareto_front.shape[-1] == 2:
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
        elif pareto_front.shape[-1] == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2])
            ax.set_xlabel('Objective 1')
            ax.set_ylabel('Objective 2')
            ax.set_zlabel('Objective 3')
        plt.title('Pareto Front')
        plt.show()
