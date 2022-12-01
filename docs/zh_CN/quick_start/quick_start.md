# 快速入门

本教程将指导您运行第一个 **OpenBox** 程序。

## 空间定义

首先，定义一个搜索空间。

```python
from openbox import sp

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])
```

在这个例子中，我们创建了一个空的搜索空间，而后向它内部添加了两个实数型（浮点型）变量。
第一个变量**x1**的取值范围是-5到10，第二个变量**x2**的取值范围是0到15。

**OpenBox**也支持其它类型的变量。
下面是定义**整型**和**类别型**变量的方法：

```python
from openbox import sp

i = sp.Int("i", 0, 100) 
kernel = sp.Categorical("kernel", ["rbf", "poly", "sigmoid"], default_value="rbf")
```

**OpenBox**的搜索空间基于**ConfigSpace**包实现。
对于更高级的用法，请参考 [ConfigSpace 官方文档](https://automl.github.io/ConfigSpace/master/index.html) 。

## 定义优化目标

第二步，定义要优化的目标函数。
注意， **OpenBox** 默认 <font color=#FF0000>**最小化**</font> 目标函数。
这里我们提供了 **Branin** 函数的例子。

```python
import numpy as np

# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return y
```

目标函数的输入是一个从搜索空间采样的配置点，输出为目标值。


## 优化

在定义了搜索空间和目标函数后，我们可以运行优化过程：

```python
from openbox import Optimizer

# Run
opt = Optimizer(
    branin,
    space,
    max_runs=50,
    surrogate_type='gp',
    time_limit_per_trial=30,
    task_id='quick_start',
)
history = opt.run()
```

这里我们创建了一个 <font color=#FF0000>**Optimizer**</font> 实例，传入目标函数 **branin** 和搜索空间 **space**。 
其余参数的含义是：

+ **num_objs=1** 和 **num_constraints=0** 表明我们的 branin 函数返回一个没有约束条件的单目标值。

+ **max_runs=50** 表示优化过程共50轮 （优化目标函数50次）。

+ **surrogate_type='gp'**： 对于数学问题，我们推荐用高斯过程 (**'gp'**) 作为贝叶斯优化的代理模型。
对于实际的问题，例如超参数优化 (HPO)，我们推荐用随机森林 (**'prf'**)。

+ **time_limit_per_trial** 为每个目标函数评估设定最大时间预算（单位：秒）。一旦评估时间超过这个限制，目标函数返回一个失败状态。

+ **task_id** 被用来区别不同优化过程。

接下来，调用 <font color=#FF0000>**opt.run()**</font> 启动优化过程。

## 可视化

在优化完成后， **opt.run()** 返回优化的历史信息。
可以通过调用 <font color=#FF0000>**print(history)**</font> 来看结果：

```python
print(history)
```

```
+-------------------------+-------------------+
| Parameters              | Optimal Value     |
+-------------------------+-------------------+
| x1                      | -3.138277         |
| x2                      | 12.254526         |
+-------------------------+-------------------+
| Optimal Objective Value | 0.398096578033325 |
+-------------------------+-------------------+
| Num Configs             | 50                |
+-------------------------+-------------------+
```

调用 <font color=#FF0000>**history.plot_convergence()**</font> 来可视化优化过程：

```python
history.plot_convergence(true_minimum=0.397887)
```

<img src="../../imgs/plot_convergence_branin.png" width="60%" class="align-center">

如果你在用 Jupyter Notebook 环境，调用 <font color=#FF0000>**history.visualize_jupyter()**</font> 来可视化每个测试：

```python
history.visualize_jupyter()
```

<img src="../../imgs/visualize_jupyter_branin.png" width="90%" class="align-center">

调用 <font color=#FF0000>**print(history.get_importance())**</font> 来输出参数的重要性：
(注意：使用该功能需要额外安装`pyrfr`包：[Pyrfr安装教程](../installation/install_pyrfr.md))

```python
print(history.get_importance())
```

```python
+------------+------------+
| Parameters | Importance |
+------------+------------+
| x1         | 0.488244   |
| x2         | 0.327570   |
+------------+------------+
```
