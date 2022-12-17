# Transfer Learning

When performing black-box optimization, users often run tasks that are similar to
previous ones. This observation can be used to speed up the current task.

OpenBox takes as input observations from $K + 1$ tasks: $D^1, ..., D^K$ 
for $K$ previous tasks and $D^T$ for the current task. 
So far, OpenBox only supports transfer learning between single-objective tasks.
Three transfer learning algorithms are supported, which are 
[RGPE](https://arxiv.org/abs/1802.02219),
[SGPR](https://dl.acm.org/doi/abs/10.1145/3097983.3098043), and 
[TransBO](https://arxiv.org/abs/2206.02663).

We have provided an example in 
[examples/transfer_learning.py](https://github.com/PKU-DAIR/open-box/blob/master/examples/transfer_learning.py).
In this tutorial, we will explain how to use transfer learning based on this example.

We first define the objective function and the search space for the current task.

```python
import numpy as np
import matplotlib.pyplot as plt
from openbox import Observation, History, Advisor, space as sp, logger

# Define config space
cs = sp.Space()
for i in range(3):
    cs.add_variable(sp.Float('x%d' % (i+1), -200, 200))

# Define objective function
def obj(config):
    x1, x2, x3 = config['x1'], config['x2'], config['x3']
    y = (x1-10)**2 + x2**2 + (x3-100)**2
    return {'objectives': [y]}
```

Then, we show how to construct a `History` class for each source task. 
We assume that we have achieved some observations on some similar tasks.
For each source task, we need to:
1) Construct `Observation`s.
2) Update the `Observation`s into the `History`.

If the source tasks are also optimized using OpenBox, the history can be obtained 
by `optimizer.get_history()` for Optimizer or `advisor.get_history()` for Advisor.

In this case, we generate one relevant source task and two irrelevant source tasks.

```python
# Generate history data for transfer learning. transfer_learning_history requires a list of History.
transfer_learning_history = list()  # type: List[History]
# 3 source tasks with 50 evaluations of random configurations each
# one task is relevant to the target task, the other two are irrelevant
num_history_tasks, num_results_per_task = 3, 50
for task_idx in range(num_history_tasks):
    # Build a History class for each source task based on previous observations.
    # If source tasks are also optimized by Openbox, you can get the History by
    # using the APIs from Optimizer or Advisor. E.g., history = advisor.get_history()
    history = History(task_id=f'history{task_idx}', config_space=cs)

    for _ in range(num_results_per_task):
        config = cs.sample_configuration()
        if task_idx == 1:  # relevant task
            y = obj(config) + 100
        else:              # irrelevant tasks
            y = np.random.random()
        # build and update observation
        observation = Observation(config=config, objectives=[y])
        history.update_observation(observation)

    transfer_learning_history.append(history)
```

To switch on transfer learning, we need to specify `transfer_learning_history` and `surrogate_type`:
+ `transfer_learning_history`: A list of History, which represents the observations from each source task.
+ `surrogate_type`: A string. Different from "auto" as shown in [Quick Start](../quick_start/quick_start),
`surrogate_type` here includes three parts.
  + The first part must be `"tlbo"`. 
  + The second part is the transfer learning algorithm, which are `"rgpe"`, `"sgpr"`, and `"topov3"`.
  + The third part is the surrogate type, e.g., `"gp"` or `"prf"`.
  
  The three parts are joint with "_".

  An example of using RGPE as the transfer learning algorithm and Gaussian Process as the surrogate is `"tlbo_rgpe_gp"`.

Here we define an `Advisor` and use the same APIs as shown in 
[examples/ask_and_tell_interface.py](https://github.com/PKU-DAIR/open-box/tree/master/examples/ask_and_tell_interface.py).

You can also define an `Optimizer` as shown in [Quick Start](../quick_start/quick_start).

```python
tlbo_advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    num_constraints=0,
    initial_trials=3,
    transfer_learning_history=transfer_learning_history,  # type: List[History]
    surrogate_type='tlbo_rgpe_gp',
    acq_type='ei',
    acq_optimizer_type='random_scipy',
    task_id='TLBO',
)

max_iter = 20
for i in range(max_iter):
    config = tlbo_advisor.get_suggestion()
    y = obj(config)
    logger.info(f'Iteration {i+1}, result: {y}')
    observation = Observation(config=config, objectives=[y])
    tlbo_advisor.update_observation(observation)
```

After optimization, we can plot the results.

```python
# show results
history = tlbo_advisor.get_history()
print(history)
history.plot_convergence()
plt.show()
```
