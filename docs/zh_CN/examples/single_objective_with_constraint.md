# 有约束条件的单目标优化

本教程介绍如何用 **OpenBox** 解决有约束条件的优化问题。

## 问题设置

首先，定义 **搜索空间** 和要最小化的 **目标函数**。
这里我们使用有约束的 **Mishra** 函数。

```python
import numpy as np
from openbox import space as sp

def mishra(config: sp.Configuration):
    X = np.array([config['x%d' % i] for i in range(2)])
    x, y = X[0], X[1]
    t1 = np.sin(y) * np.exp((1 - np.cos(x))**2)
    t2 = np.cos(x) * np.exp((1 - np.sin(y))**2)
    t3 = (x - y)**2

    result = dict()
    result['objectives'] = [t1 + t2 + t3, ]
    result['constraints'] = [np.sum((X + 5)**2) - 25, ]
    return result

params = {
    'float': {
        'x0': (-10, 0, -5),
        'x1': (-6.5, 0, -3.25)
    }
}
space = sp.Space()
space.add_variables([
    sp.Real(name, *para) for name, para in params['float'].items()
])
```

在评估后，目标函数需要返回一个 `dict` **(推荐)**
其中的结果包含：

+ `'objectives'`：一个 **要被最小化目标值** 的 **列表/元组**。
在这个例子中，我们只有一个目标，所以这个元组只包含一个值。

+ `'constraints'`：一个含有 **约束值** 的 **列表/元组**。
非正的约束值 (**"<=0"**) 表示可行。


## 优化

在定义了搜索空间和目标函数后，我们按如下方式运行优化过程：

```python
from openbox import Optimizer

opt = Optimizer(
    mishra,
    space,
    num_constraints=1,
    num_objectives=1,
    surrogate_type='gp',                # try using 'auto'!
    acq_optimizer_type='random_scipy',  # try using 'auto'!
    max_runs=50,
    task_id='soc',
    # Have a try on the new HTML visualization feature!
    # visualization='advanced',   # or 'basic'. For 'advanced', run 'pip install "openbox[extra]"' first
    # auto_open_html=True,        # open the visualization page in your browser automatically
)
history = opt.run()
```

这里我们创建一个 `Optimizer` 实例，并传入目标函数和搜索空间。
其它的参数是：

+ `num_objectives=1` 和 `num_constraints=1` 表示我们的函数返回一个有约束条件的单目标值。

+ `max_runs=50` 表示优化过程进行50轮（优化目标函数50次）。

+ `task_id` 用来识别优化过程。

+ `visualization`: `'none'`， `'basic'` 或 `'advanced'`。
详见 {ref}`可视化网页 <visualization/visualization:可视化网页>`。

+ `auto_open_html`: 是否自动在浏览器中打开可视化网页。
详见 {ref}`可视化网页 <visualization/visualization:可视化网页>`。

然后，调用 `opt.run()` 启动优化过程。

## 可视化

在优化完成后，`opt.run()` 会返回优化的历史过程。或者你可以调用 `opt.get_history()` 来获得优化历史。
接下来，调用 `print(history)` 来查看结果：

```python
history = opt.get_history()
print(history)
```

```
+-------------------------+---------------------+
| Parameters              | Optimal Value       |
+-------------------------+---------------------+
| x0                      | -3.172421           |
| x1                      | -1.506397           |
+-------------------------+---------------------+
| Optimal Objective Value | -105.72769850551406 |
+-------------------------+---------------------+
| Num Configs             | 50                  |
+-------------------------+---------------------+
```

调用 `history.plot_convergence()` 来可视化优化过程：

```python
import matplotlib.pyplot as plt
history.plot_convergence(true_minimum=-106.7645367)
plt.show()
```

<img src="../../imgs/plot_convergence_mishra.png" width="60%" class="align-center">

<font color=#FF0000>(新功能!)</font>
调用 `history.visualize_html()` 来显示可视化网页。
对于 `show_importance` 和 `verify_surrogate`，需要先运行 `pip install "openbox[extra]"`。
详细说明请参考 {ref}`可视化网页 <visualization/visualization:可视化网页>`。

```python
history.visualize_html(open_html=True, show_importance=True,
                       verify_surrogate=True, optimizer=opt)
```

<img src="../../imgs/visualization/html_example_soc.jpg" width="80%" class="align-center">
