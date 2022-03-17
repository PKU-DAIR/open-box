# LightGBM调参：使用OpenBox开源黑盒优化系统

<center class="half">
  <img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/logo.png" width="300"/><img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/tuning/LightGBM_logo_black_text.svg" width="300"/>
</center>

## 简介

本文将介绍如何使用OpenBox开源黑盒优化系统对LightGBM模型进行超参数优化。

OpenBox是针对黑盒优化设计的一套开源系统（项目地址：<https://github.com/PKU-DAIR/open-box>），以贝叶斯优化为基础，高效求解黑盒优化问题。针对超参数优化——黑盒优化的典型问题，OpenBox展现出了优异的性能，可以在更短的时间内取得更好的模型性能结果。

OpenBox使用场景广泛，不仅支持传统的单目标黑盒优化，还支持多目标优化、带约束条件优化、多种参数类型、迁移学习、分布式并行验证、多精度优化等。除了本地安装与优化调用，OpenBox还提供在线优化服务，用户可通过网页可视化监控并管理优化过程，也可以部署私有优化服务。下面我们将介绍如何在本地使用OpenBox系统对LightGBM模型调参。

## LightGBM超参数优化教程

LightGBM是基于决策树的高性能梯度提升框架，在使用时，用户需要指定模型中的一些超参数，包括学习率、树最大叶子节点数、特征采样比例等。尽管LightGBM为超参数提供了默认值，但通过调节超参数，我们可以使模型达到更好的性能。手动调节多个超参数将耗费大量验证资源与人力，使用OpenBox系统调参可以减少人工参与，并且在更少的验证次数内高效搜索到更好的结果。

在使用OpenBox进行优化之前，我们需要定义任务搜索空间（即超参数空间）和优化目标函数。OpenBox对LightGBM模型的超参数空间和目标函数进行了封装，用户可通过以下代码便捷调用（定义目标函数需提供训练与验证数据）：

```python
from openbox import get_config_space, get_objective_function
config_space = get_config_space('lightgbm')
# please prepare your data (x_train, x_val, y_train, y_val) first
objective_function = get_objective_function('lightgbm', x_train, x_val, y_train, y_val)
```

为了更好地展示任务定义细节，满足您的自定义需求，下面我们将分别介绍LightGBM超参数空间和目标函数定义方法。您也可以直接跳转至“执行优化”小节，查看如何使用OpenBox对LightGBM执行超参数优化。

### 定义超参数空间

首先，我们定义超参数空间。在这个例子中，我们的超参数空间包含7个超参数。由于最大叶子数（"num_leaves"）在一定程度上可以对树深进行控制，因此我们将最大树深（"max_depth"）设置为常量（Constant）。

```python
from openbox import sp

def get_config_space():
    cs = sp.Space()
    n_estimators = sp.Int("n_estimators", 100, 1000, default_value=500, q=50)
    num_leaves = sp.Int("num_leaves", 31, 2047, default_value=128)
    max_depth = sp.Constant('max_depth', 15)
    learning_rate = sp.Real("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    min_child_samples = sp.Int("min_child_samples", 5, 30, default_value=20)
    subsample = sp.Real("subsample", 0.7, 1, default_value=1, q=0.1)
    colsample_bytree = sp.Real("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
    cs.add_variables([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample, colsample_bytree])
    return cs

config_space = get_config_space()
```

### 定义目标函数

接下来，我们定义优化目标函数，函数输入为模型超参数，返回值为模型平衡错误率。注意：

+ OpenBox对目标向最小化方向优化。

+ 为支持多目标与带约束优化场景，返回值为字典形式，且优化目标以元组或列表表示。（单目标无约束场景下也可以返回单值）

目标函数内部包含利用训练集训练LightGBM模型、利用验证集预测并计算平衡错误率的过程。这里我们使用LightGBM提供的sklearn接口LGBMClassifier。

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier

# prepare your data
X, y = load_digits(return_X_y=True)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

def objective_function(config):
    # convert Configuration to dict
    params = config.get_dictionary()

    # fit model
    model = LGBMClassifier(**params)
    model.fit(x_train, y_train)

    # predict and calculate loss
    y_pred = model.predict(x_val)
    loss = 1 - balanced_accuracy_score(y_val, y_pred)  # OpenBox minimizes the objective

    # return result dictionary
    result = dict(objs=(loss, ))
    return result
```

### 执行优化

定义好任务和目标函数以后，就可以调用OpenBox优化框架Optimizer执行优化。我们设置优化轮数（max_runs）为100，代表将对LightGBM模型调参100轮。每轮最大验证时间（time_limit_per_trial）设置为180秒，超时的任务将被终止。优化结束后，可以打印优化结果。

```python
from openbox import Optimizer
opt = Optimizer(
    objective_function,
    config_space,
    max_runs=100,
    time_limit_per_trial=180,
    surrogate_type='prf',
    task_id='tuning_lightgbm',
)
history = opt.run()
```

打印优化结果如下：

```python
print(history)

+------------------------------------------------+
| Parameters              | Optimal Value        |
+-------------------------+----------------------+
| colsample_bytree        | 0.800000             |
| learning_rate           | 0.018402             |
| max_depth               | 15                   |
| min_child_samples       | 15                   |
| n_estimators            | 200                  |
| num_leaves              | 723                  |
| subsample               | 0.800000             |
+-------------------------+----------------------+
| Optimal Objective Value | 0.022305877305877297 |
+-------------------------+----------------------+
| Num Configs             | 100                  |
+-------------------------+----------------------+
```

我们可以绘制收敛曲线，进一步观察结果。在Jupyter Notebook环境下，还可以查看HiPlot可视化结果。

```python
history.plot_convergence()
history.visualize_jupyter()
```

<center class="half">
  <img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/plot_convergence_hpo.png" width="300"/><img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/visualize_jupyter_hpo.png" width="300"/>
</center>


上面左图为模型最优错误率随验证次数变化曲线，右图为反映优化历史中超参数与结果关系的HiPlot可视化图表。

系统还集成了超参数敏感度分析功能，依据此次任务分析超参数重要性如下：

```python
print(history.get_importance())

+--------------------------------+
| Parameters        | Importance |
+-------------------+------------+
| learning_rate     | 0.293457   |
| min_child_samples | 0.101243   |
| n_estimators      | 0.076895   |
| num_leaves        | 0.069107   |
| colsample_bytree  | 0.051856   |
| subsample         | 0.010067   |
| max_depth         | 0.000000   |
+-------------------+------------+
```

在本任务中，对模型性能影响最大的前三个超参数分别为learning_rate、min_child_samples和n_estimators。

此外，OpenBox系统还支持用字典的形式定义任务，如果您有兴趣了解，欢迎访问我们的[教程文档](https://open-box.readthedocs.io)。

## OpenBox性能实验结果

OpenBox系统对于超参数优化问题有着优异的表现，我们实验比对了各超参数优化（黑盒优化）系统性能。下图为各系统在25个数据集上，对LightGBM模型进行超参数优化后的排名：

<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/ranking_lgb_7.svg" style="zoom:30%;" />

在超参数优化问题上，OpenBox系统性能超过了现有系统。

## 总结

本文介绍了使用OpenBox系统在本地对LightGBM模型进行超参数优化的方法，并展示了系统在超参数优化问题上的性能实验结果。

如果您有兴趣了解OpenBox的更多使用方法（如多目标、带约束条件场景，并行验证，服务使用等），欢迎阅读我们的教程文档：<https://open-box.readthedocs.io>。

OpenBox项目已在Github开源，项目地址：<https://github.com/PKU-DAIR/open-box> 。欢迎更多开发者参与我们的开源项目。

