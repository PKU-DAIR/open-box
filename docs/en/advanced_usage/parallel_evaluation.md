# Parallel and Distributed Evaluation

Most proposed Bayesian optimization (BO) approaches only allow the exploration of the search space to occur 
sequentially. To fully utilize computing resources in a parallel infrastructure, **OpenBox** provides a 
mechanism for distributed parallelization, where multiple configurations can be evaluated concurrently across workers. 

Two parallel settings are considered:

<img src="../../imgs/parallel_bo.svg" width="90%" class="align-center">

1) **Synchronous parallel setting (left)**. The worker pulls a new configuration from the suggestion server to evaluate 
until all the workers have finished their last evaluations.

2) **Asynchronous parallel setting (right)**. The worker pulls a new configuration when the previous evaluation is completed.

**OpenBox** proposes a local penalization based parallelization mechanism, the goal of which is to sample new 
configurations that are promising and far enough from the configurations being evaluated by other workers.
This mechanism can handle the well-celebrated exploration vs. exploitation trade-off, and meanwhile prevent workers 
from exploring similar configurations.

In this tutorial, we illustrate how to optimize a problem in parallel manner on your local machine with **OpenBox**.

## Problem Setup

First, **define configuration space** to search and **define objective function**
to <font color=#FF0000>**minimize**</font>. Here we use the **Branin** function.

```python
import numpy as np
from openbox import space as sp

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
    return {'objectives': [y]}
```

If you are not familiar with the problem setup, please refer to 
{ref}`Quick Start Tutorial <quick_start/quick_start:quick start>`.

## Parallel Evaluation on Local

This time we use <font color=#FF0000>**ParallelOptimizer**</font> to optimize the objective function 
in a parallel manner on your local machine.

```python
from openbox import ParallelOptimizer

# Parallel Evaluation on Local Machine
opt = ParallelOptimizer(
    branin,
    space,
    parallel_strategy='async',
    batch_size=4,
    batch_strategy='default',
    num_objectives=1,
    num_constraints=0,
    max_runs=50,
    surrogate_type='gp',
    task_id='parallel_async',
)
history = opt.run()
```

In addition to **objective_function** and **space** being passed to **ParallelOptimizer**, 
the other parameters are as follows:

+ **parallel_strategy='async' / 'sync'** sets whether the parallel evaluation is performed asynchronously or synchronously.
We suggest using **'async'** because it makes better use of resources and achieves better performance than **'sync'**.

+ **batch_size=4** sets the number of parallel workers.

+ **batch_strategy='default'** sets the strategy on how to make multiple suggestions at the same time.
We suggest using **'default'** for stable performance.

+ **num_objectives=1** and **num_constraints=0** indicates that our function returns a single objective value with no constraint. 

+ **max_runs=100** means the optimization will take 100 rounds (optimizing the objective function 100 times). 

+ **surrogate_type='gp'**. For mathematical problem, we suggest using Gaussian Process (**'gp'**) as Bayesian surrogate
model. For practical problems such as hyperparameter optimization (HPO), we suggest using Random Forest (**'prf'**).

+ **task_id** is set to identify the optimization process.

After optimization, call <font color=#FF0000>**print(opt.get_history())**</font> to see the result:

```python
print(opt.get_history())
```

```
+----------------------------------------------+
| Parameters              | Optimal Value      |
+-------------------------+--------------------+
| x1                      | -3.138286          |
| x2                      | 12.292733          |
+-------------------------+--------------------+
| Optimal Objective Value | 0.3985991718620365 |
+-------------------------+--------------------+
| Num Configs             | 100                |
+-------------------------+--------------------+
```

## Distributed Evaluation

**OpenBox** provides an efficient way to perform distributed optimization.

First, start the master node with the <font color=#FF0000>**DistributedOptimizer**</font>.
We use the branin objective function defined previously.

```python
from openbox import DistributedOptimizer
from openbox.utils.platform import get_platform

# Distributed Evaluation
n_workers = 4
ip = "127.0.0.1" if get_platform() == 'Windows' else ''
opt = DistributedOptimizer(
    branin,
    space,
    parallel_strategy='async',
    batch_size=n_workers,
    batch_strategy='default',
    num_objectives=1,
    num_constraints=0,
    max_runs=50,
    surrogate_type='gp',
    task_id='distributed_opt',
    ip=ip,
    port=13579,
    authkey=b'abc',
)
history = opt.run()
```

In addition to **objective_function** and **space** being passed to **DistributedOptimizer**, 
the other parameters are as follows:

+ **port**: network port of optimizer on master node.

+ **authkey**: authorization key for worker to connect the master.

+ **parallel_strategy='async' / 'sync'** sets whether the parallel evaluation is performed asynchronously or synchronously.
We suggest using **'async'** because it makes better use of resources and achieves better performance than **'sync'**.

+ **batch_size=4** sets the number of parallel workers.

+ **batch_strategy='default'** sets the strategy on how to make multiple suggestions at the same time.
We suggest using **'default'** for stable performance.

+ **num_objectives=1** and **num_constraints=0** indicates that our function returns a single objective value with no constraint. 

+ **max_runs=100** means the optimization will take 100 rounds (optimizing the objective function 100 times). 

+ **surrogate_type='gp'**. For mathematical problem, we suggest using Gaussian Process (**'gp'**) as Bayesian surrogate
model. For practical problems such as hyperparameter optimization (HPO), we suggest using Random Forest (**'prf'**).

+ **task_id** is set to identify the optimization process.

Next, start the worker nodes to receive jobs from master and evaluate configurations.
In addition to the objective function, please specify **ip** of master node and **port**, **authkey** 
you set when starting the optimizer.

```python
from openbox import DistributedWorker

worker = DistributedWorker(branin, ip="127.0.0.1", port=13579, authkey=b'abc')
worker.run()
```

After optimization, call <font color=#FF0000>**print(opt.get_history())**</font> on master node to see the result:

```python
print(opt.get_history())
```

```
+----------------------------------------------+
| Parameters              | Optimal Value      |
+-------------------------+--------------------+
| x1                      | -3.138286          |
| x2                      | 12.292733          |
+-------------------------+--------------------+
| Optimal Objective Value | 0.3985991718620365 |
+-------------------------+--------------------+
| Num Configs             | 100                |
+-------------------------+--------------------+
```
