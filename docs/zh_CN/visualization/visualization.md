# 可视化网页

<font color=#FF0000>(新功能)</font> **OpenBox** 现提供可视化网页
以便用户查看和分析优化过程。

本教程将介绍如何使用可视化网页，并帮助理解**OpenBox**中的可视化结果。

## 如何使用可视化网页

我们假定您已经知道如何在**OpenBox**中设置问题，如果您对此不熟悉，请参考[快速入门教程](../quick_start/quick_start)。

这里我们将基于[有约束条件的多目标黑盒优化](../examples/multi_objective_with_constraint)中的例子可视化优化过程。


### 在优化前开启可视化

开启可视化网页只需在定义`Optimizer`时设置`visualization` = `basic` or `advanced`。

```python
from openbox import Optimizer
opt = Optimizer(
    ..., 
    visualization='advanced',  # or 'basic'. For 'advanced', run 'pip install lightgbm shap' first
    task_id='example_task',
    logging_dir='logs',
)
history = opt.run()
```

`visualization`有三个选项：
+ **'none'**: 运行任务时不使用可视化。
没有新的文件生成。更适合运行大规模试验。
+ **'basic'**: 运行任务时使用基本可视化，
包括针对目标值和约束值的基本图像。
+ **'advanced'**: 让可视化包含高级功能，
包括代理模型拟合分析和超参数重要性分析。

**<font color=#FF0000>注：</font>** 使用超参数重要性分析，需要安装格外的包
`shap`和`lightgbm`（先运行`pip install shap lightgbm`）。

一旦`Optimizer`被初始化，一个HTML页面会在`${logging_dir}/history/${task_id}/`中生成。
然后在浏览器中打开这个HTML页面，就可以看到优化过程的可视化。

在优化过程中，您可以点击`Refresh`按钮更新可视化结果。

### 在优化后开启可视化

如果您忘记了在`Optimizer`中设置可视化，没有关系，
您仍然可以在优化结束后开启可视化：
```python
history = opt.get_history()
history.visualize_html(
    show_importance=True,
    verify_surrogate=True,
    optimizer=opt,
)
```

这样一个HTML页面会在`${logging_dir}/history/${task_id}/`中生成。

同时请注意，如果`show_importance=True`，需要安装格外的包
`shap`和`lightgbm`（先运行`pip install shap lightgbm`）。


## 基础可视化

### 1 目标值函数

#### 1.1 目标值图像

这个示例展示在优化中每一个建议配置的目标值。

对于**有约束条件的问题**, 
满足约束的配置将被展示为圆形 <font color=#0000FF>$\bigcirc$</font>，
否则为三角形 <font color=#0000FF>$\triangle$</font>。

<img src="../../imgs/visualization/obj_value.png" width="80%" class="align-center">

<br>

#### 1.2 约束值图像

这部分可视化只适用于**有约束条件的问题**。

这个示例展示在优化中每一个建议配置的约束值。
默认情况下，非正的约束值（**"<=0"**）表示可行。

<img src="../../imgs/visualization/cons_value.png" width="80%" class="align-center">

<br>

#### 1.3 平行坐标图

这个示例展示在每个独立观察中的参数值和目标值。

<img src="../../imgs/visualization/parallel.png" width="80%" class="align-center">

<br>

### 2 多目标
这部分可视化只适用于**多目标问题**。

在多目标问题中，由于不知道哪个目标更重要，我们将寻找帕累托最优解集合。
一个帕累托最优解指的是，在不使得至少一个其他的目标变坏的情况下，这个解无法在任何一个目标上被改进。

所有的帕累托最优解组成了帕累托前沿。
我们的目标是去最大化从一个参考点到帕累托前沿的超体积。

#### 2.1 帕累托前沿

帕累托前沿可视化只适用于**多目标问题**。

帕累托前沿将被展示为曲线（2个目标时）或曲面（3个目标时）。

对于**有约束条件的问题**,
满足约束的配置将被展示为圆形 <font color=#0000FF>$\bigcirc$</font>，
否则为三角形 <font color=#0000FF>$\triangle$</font>。

<img src="../../imgs/visualization/pareto_front.png" width="80%" class="align-center">

<br>

#### 2.2 帕累托前沿超体积

这个示例展示每一轮中的由帕累托前沿包围的超体积值。

<img src="../../imgs/visualization/pareto_hypervolume.png" width="80%" class="align-center">

<br>

### 3 历史配置

这个表格记录了每一轮观察的信息。
由于空间有限无法显示所有信息（如所有的参数值），
您可以点击数据旁的 **"..."** 查看详情。

<img src="../../imgs/visualization/history.png" width="80%" class="align-center">

<br>

## 高级可视化

### 1 代理模型

在黑盒优化中，代理模型被训练去拟合配置和目标值间的关系。
我们将代理模型可视化以展现其性能。

#### 1.1 预测目标值

这个示例展示真值和预测目标值之间的关系（基于交叉验证）。
X轴是代理模型预测的目标值，Y轴是真值。
点越和线y=x接近, 说明代理模型的泛化能力越好。

<img src="../../imgs/visualization/surrogate_obj.png" width="80%" class="align-center">

<br>

#### 1.2 预测目标值排序

回顾黑盒优化的目标只是找到一个配置去最小化目标，而非精确预测每个特定配置的真值。
这里我们提供一个排序图像，这个图像和*预测目标值*图很类似。
我们基于配置的预测目标值和真值进行排序。
X轴是代理模型预测的目标值排序，Y轴是真值排序。
点越和线y=x接近, 说明代理模型的泛化能力越好。

<img src="../../imgs/visualization/surrogate_obj_rank.png" width="80%" class="align-center">

<br>

#### 1.3 预测约束值

这个图像只适用于**有约束条件的问题**。

除了目标值，我们也可以用代理模型去预测约束值。
这个图像和*预测目标值*图很类似，除了这里我们预测的是约束值。

<img src="../../imgs/visualization/surrogate_cons.png" width="80%" class="align-center">

<br>

### 2 参数重要性

我们使用**SHAP** (SHapley Additive exPlanations) 去估计参数重要性。
关于SHAP的更多信心，请参考[**SHAP documentation**](https://shap.readthedocs.io/en/latest/)。

#### 2.1 总体参数重要性

这个示例展示每个参数对于目标值的重要性。
重要值越高，这个参数对目标值的影响越大，无论是正影响还是负影响。

<img src="../../imgs/visualization/importance_obj.png" width="80%" class="align-center">

<br>

#### 2.2 对于约束值的总体参数重要性

这个示例展示每个参数对于约束值的重要性。
重要值越高，这个参数对约束值的影响越大，无论是正影响还是负影响。

<img src="../../imgs/visualization/importance_cons.png" width="80%" class="align-center">

<br>

#### 2.3 单个参数重要性

这个示例展示目标值如何依赖于某个特定参数。 
X轴是参数值，Y轴是对应的SHAP值。
SHAP值的绝对大小体现了影响程度，正值表示正相关。
您可以通过点击在上方的标签去切换参数。

对于**多目标问题**，您可以在图像上方的下拉框内选择特定目标。

<img src="../../imgs/visualization/single_obj.png" width="80%" class="align-center">

<br>

#### 2.4 对于约束值的单个参数重要性

这个示例展示约束值如何依赖于某个特定参数。 
X轴是参数值，Y轴是对应的SHAP值。
SHAP值的绝对大小体现了影响程度，正值表示正相关。
您可以通过点击在上方的标签去切换参数。

如果有**不止一个约束**，您可以在图像上方的下拉框内选择特定约束。

<img src="../../imgs/visualization/single_cons.png" width="80%" class="align-center">

<br>

