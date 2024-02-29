# Early Stopping

In this tutorial, we will show how to use the Early Stopping (ES) algorithm in **OpenBox**.
In OpenBox, the Early Stopping algorithm works by monitoring the optimization process, 
and ending the optimization early when specific conditions are met, 
thereby saving computational resources.
Currently, Early Stopping is only applicable to single-objective optimization.

## Early Stopping Strategies

The Early Stopping algorithm in OpenBox is primarily based on two strategies:

1. **No Improvement Rounds**:
   Early stopping is triggered if no improvement is observed over a certain 
   number of consecutive optimization rounds.
2. **Improvement Threshold**:
   Early stopping is triggered if the Expected Improvement (EI) is less than 
   a specific threshold, indicating that the expected improvement under the 
   current circumstances is not considered sufficient to continue optimization.
   Note that the acquisition function type must be set to EI (`acq_type='ei'`) 
   when using this strategy.

## How to use

When creating `Optimizer` or `Advisor` class, 
enable the early stopping strategy by setting `early_stop=True`.

After enabling, pass a dictionary with parameters to `early_stop_kwargs` to 
configure additional options for early stopping:

```python
opt = Optimizer(
    ...,
    early_stop=True,
    early_stop_kwargs=dict(
        min_iter=10,
        max_no_improvement_rounds=10,
        min_improvement_percentage=0.05,
    ),
)
```

The parameters are as follows:

- `min_iter` (int): The minimum number of iterations before considering early stopping. 
  This ensures the algorithm has enough time to explore the configuration space before considering stopping. 
  The default value is 10, and the range should be greater than 0.
- `max_no_improvement_rounds` (int): The maximum allowed number of rounds without improvement, 
  used for the no improvement rounds early stopping strategy.
  If no improvement is observed over several rounds, early stopping is triggered.
  The default value is 10. If set to 0, this strategy is not enabled.
- `min_improvement_percentage` (float): The minimum expected improvement percentage, 
  used for the improvement threshold early stopping strategy. If the expected improvement is less 
  than `min_improvement_percentage * (current best objective value - default objective value)`, early stopping 
  is triggered. The default value is 0.05. If set to 0, this strategy is not enabled.

## LightGBM Tuning Example

In the following example, we tune the parameters of a LightGBM model and enable the early stopping strategy.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from openbox import get_config_space, get_objective_function
from openbox import Optimizer

# prepare your data
X, y = load_digits(return_X_y=True)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# get config_space and objective_function
config_space = get_config_space('lightgbm')
objective_function = get_objective_function('lightgbm', x_train, x_val, y_train, y_val)

opt = Optimizer(
    objective_function,
    config_space,
    max_runs=100,
    surrogate_type='prf',
    task_id='tuning_lightgbm',
    early_stop=True,
    early_stop_kwargs=dict(
        min_iter=10,
        max_no_improvement_rounds=30,
        min_improvement_percentage=0.05,
    ),
    random_state=1,
)
history = opt.run()
print(history)
```

The output of early stopping is as follows:

```
[Early Stop] EI less than the threshold! min_improvement_percentage=0.05, 
default_obj=0.03334620334620342, best_obj=0.022310167310167328, 
threshold=0.0005518018018018045, max_EI=[0.00051861]

Early stop triggered at iter 41!
```

This indicates that at iteration 41, early stopping triggered because 
the expected improvement of candidate configuration was less than the set threshold.

We can compare the convergence curves with and without early stopping. 
In this early stopping configuration, optimization was effectively stopped early.

<img src="../../imgs/es_percent_convergence.png" width="60%" class="align-center">

Different early stopping options will impact the stopping round and the effect. 
For example, with `max_no_improvement_rounds=20`, `min_improvement_percentage=0.02` 
(with other configurations the same as above), the output of early stopping is as follows:

```
[Early Stop] No improvement over 21 rounds!
Early stop triggered at iter 59!
```

This indicates that at iteration 59, early stopping triggered because the objective
function did not improve over the threshold number of rounds.

Comparing the convergence curve:

<img src="../../imgs/es_round_convergence.png" width="60%" class="align-center">
