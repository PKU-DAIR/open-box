# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Optimizer, sp


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
    result['objs'] = [f1, f2]
    return result


if __name__ == "__main__":
    # search space
    space = sp.Space()
    x1 = sp.Real("x1", 0, 1)
    x2 = sp.Real("x2", 0, 1)
    space.add_variables([x1, x2])

    # provide reference point if using EHVI method
    ref_point = [18.0, 6.0]

    # run
    opt = Optimizer(
        BraninCurrin,
        space,
        num_objs=2,
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
