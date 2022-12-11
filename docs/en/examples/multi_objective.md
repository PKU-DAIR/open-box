# Multi-Objective Black-box Optimization

In this tutorial, we will introduce how to optimize multi-objective problems with **OpenBox**.

## Problem Setup

We use the multi-objective problem ZDT2 with three input dims in this example. As ZDT2 is a built-in function, 
its search space and objective function are wrapped as follows:

```python
from openbox.benchmark.objective_functions.synthetic import ZDT2

dim = 3
prob = ZDT2(dim=dim)
```

```python
import numpy as np
from openbox import space as sp
params = {'x%d' % i: (0, 1) for i in range(1, dim+1)}
space = sp.Space()
space.add_variables([sp.Real(k, *v) for k, v in params.items()])

def objective_function(config: sp.Configuration):
    X = np.array(list(config.get_dictionary().values()))
    f_0 = X[..., 0]
    g = 1 + 9 * X[..., 1:].mean(axis=-1)
    f_1 = g * (1 - (f_0 / g)**2)

    result = dict()
    result['objectives'] = np.stack([f_0, f_1], axis=-1)
    return result
```

After evaluation, the objective function returns a <font color=#FF0000>**dict (Recommended)**.</font>
The result dictionary should contain:

+ **'objectives'**: A **list/tuple** of **objective values (to be minimized)**. 
In this example, we have two objectives so the tuple contains two values.

+ **'constraints**': A **list/tuple** of **constraint values**.
If the problem is not constrained, return **None** or do not include this key in the dict.
Non-positive constraint values (**"<=0"**) imply feasibility.

## Optimization

```python
from openbox import Optimizer
opt = Optimizer(
    prob.evaluate,
    prob.config_space,
    num_objectives=prob.num_objectives,
    num_constraints=0,
    max_runs=50,
    surrogate_type='gp',
    acq_type='ehvi',
    acq_optimizer_type='random_scipy',
    initial_runs=2*(dim+1),
    init_strategy='sobol',
    ref_point=prob.ref_point,
    time_limit_per_trial=10,
    task_id='mo',
    random_state=1,
)
history = opt.run()
```

Here we create a <font color=#FF0000>**Optimizer**</font> instance, and pass the objective function 
and the search space to it. 
The other parameters are:

+ **num_objectives** and **num_constraints** set how many objectives and constraints the objective function will return.
In this example, **num_objectives=2**.

+ **max_runs=50** means the optimization will take 50 rounds (optimizing the objective function 50 times). 

+ **surrogate_type='gp'**. For mathematical problem, we suggest using Gaussian Process (**'gp'**) as Bayesian surrogate
model. For practical problems such as hyperparameter optimization (HPO), we suggest using Random Forest (**'prf'**).

+ **acq_type='ehvi'**. Use **EHVI(Expected Hypervolume Improvement)** as Bayesian acquisition function. For problems with more than 3 objectives, please
use **MESMO('mesmo')** or **USEMO('usemo')**.

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
Call <font color=#FF0000>**opt.get_history().plot_pareto_front()**</font> to plot the pareto front.
Please note that `plot_pareto_front` only works when the number of objectives is 2 or 3.

```python
import matplotlib.pyplot as plt

history = opt.get_history()
# plot pareto front
if history.num_objectives in [2, 3]:
    history.plot_pareto_front()  # support 2 or 3 objectives
    plt.show()
```

<img src="../../imgs/plot_pareto_front_zdt2.png" width="60%" class="align-center">

Then plot the hypervolume difference during the optimization compared to the ideal pareto front.

```python
# plot hypervolume
hypervolume = opt.get_history().hv_data
log_hv_diff = np.log10(prob.max_hv - np.asarray(hypervolume))
plt.plot(log_hv_diff)
plt.xlabel('Iteration')
plt.ylabel('Log Hypervolume Difference')
plt.show()
```

<img src="../../imgs/plot_hypervolume_zdt2.png" width="60%" class="align-center">
