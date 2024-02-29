# 自动化算法选择

Since a large number of Bayesian optimization algorithms are proposed,
users may find it difficult to choose the proper algorithm for their tasks.
**OpenBox** provides an automatic algorithm selection mechanism to choose the proper 
optimization algorithm for a given optimization task.

This document provides a brief introduction to the algorithm selection mechanism,
including the usage of the mechanism and the algorithm selection criteria.


## Usage

To use the automatic algorithm selection mechanism,
set the following options to `'auto'` in `Advisor` or `Optimizer`:
+ `surrogate_type='auto'`
+ `acq_type='auto'`
+ `acq_optimizer_type='auto'`

\*By default, the algorithm selection mechanism is enabled.

For example:
```python
from openbox import Advisor
advisor = Advisor(
    ...,
    surrogate_type='auto',
    acq_type='auto',
    acq_optimizer_type='auto',
)
```

After initialization, a log message will be printed to indicate the selected algorithms:
```
[BO auto selection]  surrogate_type: gp. acq_type: ei. acq_optimizer_type: random_scipy.
```


## Algorithm Selection Criteria

The algorithm selection mechanism is based on the characteristics of the problem, 
such as the dimensionality, the types of hyperparameters, and the number of objectives. 
It is designed to provide consistent performance for different problems.

The criteria for algorithm selection are obtained from practical experience or experimental results.

### For Surrogate Model

Gaussian Process (GP, `'gp'`) vs. Probabilistic Random Forest (PRF, `'prf'`):
+ GP performs very well on mathematical functions.
+ GP performs well for space with continuous hyperparameters.
+ PRF is better if the space is full of categorical hyperparameters.
+ GP is not suitable for high-dimensional problems.
+ PRF can be used for high-dimensional problems.
+ Computational cost: GP is $O(n^3)$ while PRF is $O(nlogn)$, where $n$ is the number of observations.

Currently, the algorithm selection mechanism selects the proper surrogate model based on the following criteria:
+ If there are 10 or more hyperparameters in the search space, `'prf'` is selected.
  (If there are 100 or more hyperparameters, random search is used instead of BO.)
+ If there are more categorical hyperparameters than continuous hyperparameters, `'prf'` is selected.
+ Otherwise, `'gp'` is selected.
+ If the model is automatically selected to be `'gp'`, and the number of observations is greater than 300,
  the model is automatically switched to `'prf'`.

### For Acquisition Function

The acquisition function is chosen based on the type of the optimization task:

+ For single-objective optimization (SO), the widely used Expected Improvement (EI, `'ei'`) is selected.
+ For single-objective optimization with constraints (SOC), the Expected Improvement with Constraints (EIC, `'eic'`) 
  is selected.
+ For multi-objective optimization (MO):
  + If `num_objectives <= 4`, the Expected Hypervolume Improvement (EHVI, `'ehvi'`) is selected.
    (The computational cost of EHVI grows exponentially with the number of objectives, 
    so it is not suitable for problems with too many objectives. Typically, the threshold is set to 4.)
  + Otherwise, the Max-value Entropy Search for Multi-Objective (MESMO, `'mesmo'`) is selected.
+ For multi-objective optimization with constraints (MOC):
  + If `num_objectives <= 4`, the Expected Hypervolume Improvement with Constraints (EHVIC, `'ehvic'`) is selected.
  + Otherwise, the Max-value Entropy Search for Multi-Objective with Constraints (MESMOC, `'mesmoc'`) is selected.

### For Acquisition Function Optimizer

Currently supported acquisition function optimizers:
+ `'local_random'`: Interleaved Local and Random Search.
+ `'random_scipy'`: Random Search and L-BFGS-B optimizer from SciPy.

The `'random_scipy'` optimizer requires all hyperparameters to be continuous currently.
It costs more time than `'local_random'` but is more effective.

The `'local_random'` optimizer is available for all scenarios.

Currently, the algorithm selection mechanism selects the proper acquisition function optimizer 
based on the following criteria:
+ If categorical hyperparameters exist in the search space, `'local_random'` is selected.
+ Otherwise, `'random_scipy'` is selected.


## Extend Mechanism of Automatic Algorithm Selection

For users who want to extend or customize the algorithm selection mechanism,
please rewrite the `algo_auto_selection` method in `Advisor`.

For example, if you want to use the Probability of improvement (PI) as the acquisition function 
for single-objective optimization if there are more than 10 hyperparameters in the search space:

```python
from openbox import Advisor, logger

class MyAdvisor(Advisor):
    def algo_auto_selection(self):
        if self.acq_type == 'auto':
            n_dim = len(self.config_space.get_hyperparameters())
            if self.num_objectives == 1 and self.num_constraints == 0 and n_dim > 10:
                self.acq_type = 'pi'
                logger.info(f'[auto selection] acq_type: {self.acq_type}')
        super().algo_auto_selection()
```
