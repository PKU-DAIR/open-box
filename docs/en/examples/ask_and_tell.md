# Ask-and-tell Interface

In this tutorial, we will introduce how to use the ask-and-tell interface `Advisor` in **OpenBox**. 

## Design Principles

**OpenBox** provides two interfaces for users to interact with the black-box optimization process:
`Optimizer` and `Advisor`.

1. `Optimizer` is a high-level interface that encapsulates the whole optimization process.
    Users only need to provide the objective function and the search space, and the `Optimizer` 
    will handle the rest. The `Optimizer` will automatically generate new configurations, run 
    the objective function, and update the observation iteratively, just via `Optimizer.run()`.
2. `Advisor` is a low-level ask-and-tell interface that provides users with more flexibility. 
    For users who want to control the optimization process manually, they can use `Advisor` to
    get configuration suggestions (via `advisor.get_suggestion()`), then run the objective
    function manually, and update the observation back (via `advisor.update_observation(observation)`).

In fact, the `Advisor` is a part of the `Optimizer`. When using the `Optimizer`, the `Optimizer` 
will get suggestion from `Advisor` and update observation to `Advisor`. There's no difference in 
performance between the two interfaces.


## Basic Workflow

The basic workflow of using `Advisor` is as follows:
```python
from openbox import Advisor, Observation
# Define Advisor
advisor = Advisor(config_space, ...)
# Loop
for i in range(num_iters):
    # Ask
    config = advisor.get_suggestion()
    # Evaluate on user-defined objective function
    y = objective_function(config)
    # Build observation
    observation = Observation(config=config, objectives=[y, ])
    # Tell
    advisor.update_observation(observation)
```

## API

Most parameters in `Advisor` are the same as those in `Optimizer`.

*The module is still in active development, and the API may change in the future.

Parameters:
- `config_space` (openbox.space.Space or ConfigSpace.ConfigurationSpace): Configuration space.
- `num_objectives` (int, default=1): Number of objectives in objective function.
- `num_constraints` (int, default=0): Number of constraints in objective function.
- `initial_trials` (int, default=3): Number of initial iterations of optimization.
- `init_strategy` (str, default='random_explore_first'): Strategy to generate configurations for initial iterations.
- `initial_configurations` (List[Configuration], optional): If provided, the initial configurations will be evaluated in initial iterations of optimization.
- `transfer_learning_history` (List[History], optional): Historical data for transfer learning.
- `rand_prob` (float, default=0.1): Probability to sample random configurations.
- `surrogate_type` (str, default='auto'): Type of surrogate model in Bayesian optimization.
- `acq_type` (str, default='auto'): Type of acquisition function in Bayesian optimization.
- `acq_optimizer_type` (str, default='auto'): Type of optimizer to maximize acquisition function.
- `ref_point` (List[float], optional): Reference point for calculating hypervolume in multi-objective problem.
- `output_dir` (str, default='logs'): Directory to save log files. If None, no log files will be saved.
- `task_id` (str, default='OpenBox'): Task identifier.
- `random_state` (int): Random seed for RNG.
- `logger_kwargs` (dict, optional): Additional keyword arguments for logger.


## Example

```python
import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, space as sp, Observation, logger

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


# Run
if __name__ == "__main__":
    advisor = Advisor(
        space,
        # surrogate_type='gp',
        surrogate_type='auto',
        task_id='ask_and_tell',
    )

    MAX_RUNS = 50
    for i in range(MAX_RUNS):
        # ask
        config = advisor.get_suggestion()
        # evaluate
        ret = branin(config)
        # tell
        observation = Observation(config=config, objectives=ret['objectives'])
        advisor.update_observation(observation)
        logger.info('\n===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))

    history = advisor.get_history()
    print(history)

    history.plot_convergence(true_minimum=0.397887)
    plt.show()
```
