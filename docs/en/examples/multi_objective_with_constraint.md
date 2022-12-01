# Multi-Objective with Constraints

In this tutorial, we will introduce how to optimize constrained multiple objectives problem with **OpenBox**.

## Problem Setup

We use constrained multi-objective problem CONSTR in this example. As CONSTR is a built-in function, 
its search space and objective function are wrapped as follows:

```python
from openbox.benchmark.objective_functions.synthetic import CONSTR

prob = CONSTR()
dim = 2
initial_runs = 2 * (dim + 1)
```

```python
import numpy as np
from openbox import sp
params = {'x1': (0.1, 10.0),
          'x2': (0.0, 5.0)}
space = sp.Space()
space.add_variables([sp.Real(k, *v) for k, v in params.items()])

def objective_funtion(config: sp.Configuration):
    X = np.array(list(config.get_dictionary().values()))

    result = dict()
    obj1 = X[..., 0]
    obj2 = (1.0 + X[..., 1]) / X[..., 0]
    result['objs'] = np.stack([obj1, obj2], axis=-1)

    c1 = 6.0 - 9.0 * X[..., 0] - X[..., 1]
    c2 = 1.0 - 9.0 * X[..., 0] + X[..., 1]
    result['constraints'] = np.stack([c1, c2], axis=-1)

    return result
```

After evaluation, the objective function returns a <font color=#FF0000>**dict (Recommended)**.</font>
The result dictionary should contain:

+ **'objs'**: A **list/tuple** of **objective values (to be minimized)**. 
In this example, we have two objectives so the tuple contains two values.

+ **'constraints**': A **list/tuple** of **constraint values**.
Non-positive constraint values (**"<=0"**) imply feasibility.

## Optimization

```python
from openbox import Optimizer
opt = Optimizer(
    prob.evaluate,
    prob.config_space,
    num_objs=prob.num_objs,
    num_constraints=prob.num_constraints,
    max_runs=100,
    surrogate_type='gp',
    acq_type='ehvic',
    acq_optimizer_type='random_scipy',
    initial_runs=initial_runs,
    init_strategy='sobol',
    ref_point=prob.ref_point,
    time_limit_per_trial=10,
    task_id='moc',
    random_state=1,
)
opt.run()
```

Here we create a <font color=#FF0000>**Optimizer**</font> instance, and pass the objective function 
and the search space to it. 
The other parameters are:

+ **num_objs** and **num_constraints** set how many objectives and constraints the objective function will return.
In this example, **num_objs=2** and **num_constraints=2**.

+ **max_runs=100** means the optimization will take 100 rounds (optimizing the objective function 100 times). 

+ **surrogate_type='gp'**. For mathematical problem, we suggest using Gaussian Process (**'gp'**) as Bayesian surrogate
model. For practical problems such as hyperparameter optimization (HPO), we suggest using Random Forest (**'prf'**).

+ **acq_type='ehvic'**. Use **EHVIC(Expected Hypervolume Improvement with Constraint)**
as Bayesian acquisition function.

+ **acq_optimizer_type='random_scipy'**. For mathematical problems, we suggest using **'random_scipy'** as
acquisition function optimizer. For practical problems such as hyperparameter optimization (HPO), we suggest
using **'local_random'**.

+ **initial_runs** sets how many configurations are suggested by **init_strategy** before the optimization loop.

+ **init_strategy='sobol'** sets the strategy to suggest the initial configurations.

+ **ref_point** specifies the reference point, which is the upper bound on the objectives used for computing
hypervolume. If using EHVI method, a reference point must be provided. In practice, the reference point can be
set 1) using domain knowledge to be slightly worse than the upper bound of objective values, where the upper bound is
the maximum acceptable value of interest for each objective, or 2) using a dynamic reference point selection strategy.

+ **time_limit_per_trial** sets the time budget (seconds) of each objective function evaluation. Once the 
evaluation time exceeds this limit, objective function will return as a failed trial.

+ **task_id** is set to identify the optimization process.

Then, <font color=#FF0000>**opt.run()**</font> is called to start the optimization process.

## Visualization

Since we optimize both objectives at the same time, we get a pareto front as the result.
Call <font color=#FF0000>**opt.get_history().get_pareto_front()**</font> to get the pareto front.

```python
import numpy as np
import matplotlib.pyplot as plt
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
```

<img src="../../imgs/plot_pareto_front_constr.png" width="60%" class="align-center">

Then plot the hypervolume difference during the optimization compared to the ideal pareto front.

```python
# plot hypervolume
hypervolume = opt.get_history().hv_data
max_hv = 92.02004226679216
log_hv_diff = np.log10(max_hv - np.asarray(hypervolume))
plt.plot(log_hv_diff)
plt.xlabel('Iteration')
plt.ylabel('Log Hypervolume Difference')
plt.show()
```

<img src="../../imgs/plot_hypervolume_constr.png" width="60%" class="align-center">
