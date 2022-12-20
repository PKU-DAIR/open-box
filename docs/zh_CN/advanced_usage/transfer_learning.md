# 迁移学习

在进行黑盒优化时，用户经常运行与以前类似的任务。这个观察可用于加速当前优化任务。

OpenBox 的以 $K + 1$ 个任务的观测作为输入: $D^1, ..., D^K$ 指 $K$ 个先前的任务， $D^T$ 指当前任务。
目前，OpenBox只支持单目标任务间的迁移学习，并提供了3个迁移学习算法，包括
[RGPE](https://arxiv.org/abs/1802.02219)，
[SGPR](https://dl.acm.org/doi/abs/10.1145/3097983.3098043)， 和
[TransBO](https://arxiv.org/abs/2206.02663)。

我们提供了一个样例程序：
[examples/transfer_learning.py](https://github.com/PKU-DAIR/open-box/blob/master/examples/transfer_learning.py).
在本教程中，我们将基于此样例程序解释如何使用迁移学习。

首先，我们为当前任务定义目标函数和搜索空间。

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

接下来，我们展示如何为每一个源任务构建一个`History`类。
我们假设我们已在相似的任务上获取了一些观察。
对于每个源任务，我们需要：
1) 构建观察 `Observation`。
2) 将观察 `Observation` 更新到历史 `History` 中。

如果源任务也是使用OpenBox优化得到的，那么我们可以使用`optimizer.get_history()`从Optimizer中获取历史，
或使用`advisor.get_history()`从Advisor中获取历史。

在这个例子里，我们生成了一个相关的源任务，和两个不相关的源任务。

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
            y = obj(config)['objectives'][0] + 100
        else:              # irrelevant tasks
            y = np.random.random()
        # build and update observation
        observation = Observation(config=config, objectives=[y])
        history.update_observation(observation)

    transfer_learning_history.append(history)
```

为了启用迁移学习，我们需要指定 `transfer_learning_history` 和 `surrogate_type`:
+ `transfer_learning_history`: 一个包含History的列表，每个History对应一个源任务。
+ `surrogate_type`: 一个字符串。不同于 {ref}`快速入门教程 <quick_start/quick_start:快速入门>` 中展示的"auto",
  这里的`surrogate_type`包含三个部分：
  + 第一个部分必须是`"tlbo"`。
  + 第二个部分是迁移学习算法，包括 `"rgpe"`， `"sgpr"`， 和 `"topov3"`。
  + 第三个部分是代理模型类型，例如 `"gp"` 或 `"prf"`。
  
  三个部分用"_"连接成一个字符串。
  
  使用RGPE作为迁移学习算法并使用高斯过程作为代理模型的例子为`"tlbo_rgpe_gp"`。

我们这里定义了一个 `Advisor` 并使用了和
[examples/ask_and_tell_interface.py](https://github.com/PKU-DAIR/open-box/tree/master/examples/ask_and_tell_interface.py)
中展示的相同API。

您也可以定义一个 `Optimizer` ，参考 {ref}`快速入门教程 <quick_start/quick_start:快速入门>`。

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
    res = obj(config)
    logger.info(f'Iteration {i+1}, result: {res}')
    observation = Observation(config=config, objectives=res['objectives'])
    tlbo_advisor.update_observation(observation)
```

优化后，我们可以展示结果。

```python
# show results
history = tlbo_advisor.get_history()
print(history)
history.plot_convergence()
plt.show()
```
