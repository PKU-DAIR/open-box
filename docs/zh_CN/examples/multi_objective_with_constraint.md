# 有约束条件的多目标黑盒优化

本教程介绍如何使用**OpenBox**解决有约束条件的多目标优化问题。


## 问题设置

本例中，我们使用带约束的多目标优化问题 CONSTR 作为例子。
OpenBox内置了CONSTR函数，其搜索空间和目标函数被包装如下：

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

在评估后，目标函数返回一个 <font color=#FF0000>**dict (Recommended)**</font>。
这个结果目录包含：

+ **'objs'**: 一个 **要被最小化目标值** 的 **列表/元组**。
在这个例子中，我们有两个目标，所以这个元组包含两个值。

+ **'constraints**': 一个含有 **约束值** 的 **列表/元组**。
 非正的约束值 (**"<=0"**) 表示可行。


## 优化

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

这里我们创建一个 <font color=#FF0000>**Optimizer**</font> 实例，并传入目标函数和搜索空间。
其它的参数是：

+ **num_objs** 和 **num_constraints** 设置目标函数将返回多少目标和约束。在这个例子中，**num_objs=2**，**num_constraints=2**。

+ **max_runs=100** 表示优化会进行100轮（优化目标函数100次）。

+ **surrogate_type='gp'** 对于数学问题，我们推荐用高斯过程 (**'gp'**) 做贝叶斯优化的替代模型。
对于实际问题，比如超参数优化（HPO）问题，我们推荐使用随机森林(**'prf'**)。

+ **acq_type='ehvic'** 用 **EHVIC(Expected Hypervolume Improvement with Constraint)** 作为贝叶斯优化的acquisition function。

+ **acq_optimizer_type='random_scipy'** 对于数学问题，我们推荐用 **'random_scipy'** 作为贝叶斯优化的acquisition function。
  对于实际问题，比如超参数优化（HPO）问题，我们推荐使用 **'local_random'** 。

+ **initial_runs** 设置在优化循环之前，**init_strategy**推荐使用的配置数量。
  
+ **init_strategy='sobol'** 设置建议初始配置的策略。

+ **ref_point** 指定参考点，它是用于计算超体积的目标的上限。
  如果使用EHVI方法，则必须提供参考点。
  在实践中，可以1）使用领域知识将参考点设置为略差于目标值的上界，其中上界是每个目标感兴趣的最大可接受值，或者2）使用动态的参考点选择策略。
  
+ **time_limit_per_trial** 为每个目标函数评估设定最大时间预算（单位：秒）。一旦评估时间超过这个限制，目标函数返回一个失败状态。

+ **task_id** 用来识别优化过程。

然后，调用 <font color=#FF0000>**opt.run()**</font> 启动优化过程。

## 可视化

由于我们同时优化了这两个目标，我们得到了一个pareto front作为结果。
调用 <font color=#FF0000>**opt.get_history().get_pareto_front()**</font> 来获取pareto front。

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

然后绘制优化过程中与理想pareto front相比的hypervolume差。

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
