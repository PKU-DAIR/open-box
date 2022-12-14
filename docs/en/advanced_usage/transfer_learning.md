# Transfer Learning

When performing black-box optimization, users often run tasks that are similar to
previous ones. This observation can be used to speed up the current task.

OpenBox takes as input observations from $K + 1$ tasks: $D^1$, ...,
$D^K$ for $K$ previous tasks and $D^T$ for the current task. 
So far, OpenBox only supports transfer learning between single-objective tasks.
Three transfer learning algorithms are supported, which are [RGPE](https://arxiv.org/abs/1802.02219), [SGPR](https://dl.acm.org/doi/abs/10.1145/3097983.3098043), and [TransBO](https://arxiv.org/abs/2206.02663).

We have provided an example `transfer_learning.py`under [Examples](https://github.com/PKU-DAIR/open-box/tree/master/examples).
In this documentation, we will explain how to use transfer learning based on this example.

In this example, we first show how to construct a History class for each source task. 
We assume that we have achieved some observations on some similar tasks.
Then, we need to 1) define an Observation and 2) update the Observation into the History.
If the source tasks are also optimized using OpenBox, the history can be obtained by `optimizer.run()` for Optimizer or `advisor.get_history()` for Advisor.

```python
cs = ConfigurationSpace()
hp1 = UniformFloatHyperparameter('x1', -5, 10)
hp2 = UniformFloatHyperparameter('x2', 0, 15)
cs.add_hyperparameters([hp1, hp2])


# The source objective functions, which are slightly different from the target function.
def branin_source_1(config):
    x1, x2 = config['x1'], config['x2']
    y = (2 * x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y


def branin_source_2(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10 + 10
    return y

# The target objective function.
def branin_target(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y


# In both source tasks, we assume we have already evaluated 50 random configurations.
num_source_samples = 50
source_configs_1 = cs.sample_configuration(num_source_samples)
source_perfs_1 = [branin_source_1(config) for config in source_configs_1]

# Build a History class for each source task based on previous observations.
# If source tasks are also optimized by Openbox, you can get the History by using the APIs from Optimizer or Advisor.
history_1 = History(task_id='history1', config_space=cs)
for i, _ in enumerate(source_configs_1):
    observation = Observation(config=source_configs_1[i], objectives=[source_perfs_1[i]])
    history_1.update_observation(observation)

source_configs_2 = cs.sample_configuration(num_source_samples)
source_perfs_2 = [branin_source_2(config) for config in source_configs_2]

history_2 = History(task_id='history2', config_space=cs)
for i, _ in enumerate(source_configs_2):
    observation = Observation(config=source_configs_2[i], objectives=[source_perfs_2[i]])
    history_2.update_observation(observation)
```

Then, to switch on transfer learning, we need to define an Advisor by setting `history_bo_data` and `surrogate_type`:
+ `history_bo_data`：A list of History, which represents the observations from each source task.
+ `surrogate_type`: A string. Different from "auto" as shown in [Quick Start](../quick_start/quick_start),
surrogate_type here includes three parts. 
The first part must be "tlbo". 
The second part is the transfer learning algorithm, which are "rgpe", "sgpr", and "topov3".
The third part is the surrogate type, e.g., "gp" or "prf".
The three parts are joint with "_".
An example of using RGPE as the transfer learning algorithm and Gaussian Process as the surrogate is "tlbo_rgpe_gp".

```python
rgpe_advisor = Advisor(cs, num_objectives=1, num_constraints=0, initial_trials=5,
                       history_bo_data=[history_1, history_2],
                       surrogate_type='tlbo_rgpe_gp', 
                       acq_type='ei', acq_optimizer_type='random_scipy')
```

Finally, we can run optimization using the same APIs as shown in `ask_and_tell_interface.py` in [Examples](https://github.com/PKU-DAIR/open-box/tree/master/examples).

```python
iteration_budget = 50
for i in range(iteration_budget):
    config = rgpe_advisor.get_suggestion()
    result = branin_target(config)
    print('The %d-th iteration: Result %f' % (i, result))
    observation = Observation(config=config, objectives=[result])
    rgpe_advisor.update_observation(observation)
```

