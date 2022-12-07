# 并行和分布式验证

大多数贝叶斯优化方法（BO）只对搜索空间进行串行的探索。
为了充分利用并行设施中的计算资源，**OpenBox**提出了一种分布式并行化的机制。
在这种机制中，多个配置可以在多个worker上并发地被验证。

我们考虑了两种并行机制：

<img src="../../imgs/parallel_bo.svg" width="90%" class="align-center">

1) **同步并行（左）**：每个worker从推荐配置中选择一个配置进行验证。直到所有worker都完成这一轮的验证后再开始下一轮验证。

2) **异步并行（右）**：每个worker从推荐配置中选择一个配置进行验证。对于每个worker，当前的验证结束后就立刻开始下一轮验证。

**OpenBox** 实现了一种基于局部惩罚的并行化机制。其目标是对有前途的新配置进行采样，这个采样要与其它worker正在验证的配置相差足够远。
这种机制可以处理众所周知的exploration与exploitation之间的权衡，同时防止工人探索类似的配置。

在本教程中，我们将演示如何使用**OpenBox**在本地计算机上以并行的方式解决优化问题。


## 问题描述

首先，定义搜索的**搜索空间**和想要<font color=#FF0000>**最小化**</font>的**目标函数**。这里我们使用**Branin**函数。

```python
import numpy as np
from openbox import sp

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
    return {'objs': (y,)}
```

如果你对这个问题描述还不熟悉，请参考我们的[快速入门教程](../quick_start/quick_start)。

## 单机并行验证

这里我们使用 <font color=#FF0000>**ParallelOptimizer**</font> 来以并行的方式在单机上优化目标函数。

```python
from openbox import ParallelOptimizer

# Parallel Evaluation on Local Machine
opt = ParallelOptimizer(
    branin,
    space,
    parallel_strategy='async',
    batch_size=4,
    batch_strategy='default',
    num_objs=1,
    num_constraints=0,
    max_runs=50,
    surrogate_type='gp',
    time_limit_per_trial=180,
    task_id='parallel_async',
)
history = opt.run()
```

除了被传递给 **ParallelOptimizer** 的 **目标函数** 和 **搜索空间**，其它的参数有：

+ **parallel_strategy='async' / 'sync'** 设置并行验证是异步的还是同步的。
我们推荐使用 **'async'** 异步并行方式，因为它能更充分地利用资源，减少空闲。 
实验表明异步并行相比 **'sync'** 同步并行达到了更好的性能。

+ **batch_size=4** 设置并行worker的数量。

+ **batch_strategy='default'** 设置如何同时提出多个建议的策略。
我们推荐使用默认参数 **'default'** 来获取稳定的性能。

+ **num_objs=1** 和 **num_constraints=0** 表明我们的函数返回一个没有约束的单目标值。

+ **max_runs=100** 表明优化过程循环100次 (优化目标函数100次). 

+ **surrogate_type='gp'** 对于数学问题，我们推荐使用高斯过程(**'gp'**)作为贝叶斯优化的替代模型。 
  对于实际问题，比如超参数优化（HPO），我们推荐使用随机森林(**'prf'**)。

+ **time_limit_per_trial** 设置每次目标函数验证的时间预算（秒）。
  一旦验证时间超过了这个限制，目标函数返回一个失败的测试。
  
+ **task_id** 指明优化任务名称。

在优化完成后, 调用 <font color=#FF0000>**print(opt.get_history())**</font> 来打印优化结果:

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


## 分布式并行验证

**OpenBox** 提供了一个便捷的方式执行分布式并行验证。

首先，在主节点上启动优化器<font color=#FF0000>**DistributedOptimizer**</font>。
这里我们仍使用前面定义的Branin函数作为目标函数。

```python
from openbox import DistributedOptimizer

# Distributed Evaluation
n_workers = 4
opt = DistributedOptimizer(
    branin,
    space,
    parallel_strategy='async',
    batch_size=n_workers,
    batch_strategy='default',
    num_objs=1,
    num_constraints=0,
    max_runs=50,
    surrogate_type='gp',
    time_limit_per_trial=180,
    task_id='distributed_opt',
    port=13579,
    authkey=b'abc',
)
history = opt.run()
```

除了被传递给 **ParallelOptimizer** 的 **目标函数** 和 **搜索空间**，其它的参数有：

+ **port**: 主节点上优化器的网络端口号。

+ **authkey**: worker连接主节点优化器所需的授权秘钥。

+ **parallel_strategy='async' / 'sync'** 设置并行验证是同步的还是异步的。
我们推荐使用 **'async'** 因为它能更充分地利用资源，并比 **'sync'** 实现了更好的性能。

+ **batch_size=4** 设置并行worker的数量。

+ **batch_strategy='default'** 设置如何同时提出多个建议的策略。
我们推荐使用默认参数 **'default'** 来获取稳定的性能。

+ **num_objs=1** 和 **num_constraints=0** 表明我们的函数返回一个没有约束的单目标值。

+ **max_runs=100** 表明优化过程循环100次 (优化目标函数100次). 

+ **surrogate_type='gp'** 对于数学问题，我们推荐使用高斯过程(**'gp'**)作为贝叶斯优化的替代模型。 
  对于实际问题，比如超参数优化（HPO），我们推荐使用随机森林(**'prf'**)。

+ **time_limit_per_trial** 设置每次目标函数验证的时间预算（秒）。
  一旦验证时间超过了这个限制，目标函数返回一个失败的测试。
  
+ **task_id** 指明优化任务名称。

接下来，启动worker节点以从主节点接收参数配置，并执行验证。
除了目标函数，在启动worker时还需要指定主节点**ip**, 端口号**port**和授权秘钥**authkey**。

```python
from openbox import DistributedWorker

worker = DistributedWorker(branin, ip="127.0.0.1", port=13579, authkey=b'abc')
worker.run()
```

在优化完成后, 在主节点上调用 <font color=#FF0000>**print(opt.get_history())**</font> 来打印优化结果:

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
