# Tuning LightGBM with OpenBox -- An Open Source Black-box Optimization System

<center class="half">
  <img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/logo.png" width="300"/><img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/tuning/LightGBM_logo_black_text.svg" width="300"/>
</center>

## Introduction

In this article, we will introduce how to tune LightGBM with OpenBox.

[OpenBox](https://github.com/PKU-DAIR/open-box) is an open-source system designed for efficient black-box optimization using Bayesian optimization. OpenBox shows excellent performance on hyperparameter tuning, which is a typical scenario of black-box optimization.

OpenBox has a wide range of usages, including black-optimization with a different number of objectives and constraints, transfer learning, parallel evaluations, multi-fidelity optimization, etc. In addition to local installation, OpenBox provides an online service. The users can monitor and manage the optimization process through web pages and deploy their optimization services privately. In the following, we will introduce how to tune LightGBM locally using OpenBox.

## Tutorial for Tuning LightGBM

LightGBM is a popular library that provides an effective and efficient gradient boosting framework based on decision tree algorithms. A LightGBM model requires the definition of various hyperparameters, including the learning rate, the subsample rate, etc. Though LightGBM provides a default value for each hyperparameter, we may achieve better performance using a certain combination of hyperparameter values. While tuning them manually requires efforts, tuning with OpenBox could efficiently find better performance under a limited budget.

Before optimizing using OpenBox, we need to define the task search space (i.e., the hyperparameter space) and the objective function. OpenBox has provided wrapped APIs for ease of use. The users can tune their LightGBM by simply writing the following three lines of code:

```python
from openbox import get_config_space, get_objective_function
config_space = get_config_space('lightgbm')
# please prepare your data (x_train, x_val, y_train, y_val) first
objective_function = get_objective_function('lightgbm', x_train, x_val, y_train, y_val)
```

To display the details of defining a task, we then describe the personalized definition of the search space and objective function as follows. You may also skip to Section Optimization to check how to tune LightGBM with OpenBox.

### Search Space

First, we define the search space. In the following example, the search space contains seven hyperparameters. Since the hyperparameter *num_leaves* could to some extent limit the depth of each decision tree, we set the hyperparameter *max_depth* as a constant.

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

### Objective Function

Then, we need to define the objective function, which takes the model hyperparameters as inputs and returns the balanced accuracy evaluated using the validation set. Note that:

+ By default, OpenBox **minimizes** the objective function. (If the users want to maximize the objective function, please return its negative value.)
+ To support multiple objectives and constraints, it is recommended to return a Python *dictionary*, and the objectives should be given in a *list* or *tuple*. (Returning a single value is ok if there's only a single objective.)

The following objective function involves two steps: 1) training a LightGBM model on the training set and 2) evaluate its balanced accuracy on the validation set. We directly use the LGBMClassifier provided by the package LightGBM.

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

### <span id = "anchor">Optimization</span>

After defining the search space and the objective function, we can use the built-in optimization framework to perform optimization. In the following example, we set *max_runs* to 100, which means OpenBox will tune LightGBM 100 times. In addition, each run is given a time limit of 180 seconds by setting *time_limit_per_trial*. 

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

After the optimization, the run history can be printed as follows:

```python
print(history)

+-------------------------+----------------------+
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

The convergence curve can be plotted for further visualization. If the code is run with Jupyter Notebook, the package HiPlot will show more interesting results:

```python
history.plot_convergence()
history.visualize_jupyter()
```

<center class="half">
  <img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/plot_convergence_hpo.png" width="300"/><img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/visualize_jupyter_hpo.png" width="300"/>
</center>


The left figure shows the best observed objective during the optimization while the right figure reflects the relationships between each hyperparameter and the objective.

In addition, Openbox has also integrated the functionality of analyzing hyperparameter importance.

```python
print(history.get_importance())

+-------------------+------------+
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

In this task, the top-3 influential hyperparameters are *learning_rate*, *min_child_samples*, and *n_estimators*.

In addition, OpenBox supports defining a task using a JSON file. For more characteristics and details about OpenBox, please refer to our [documentation](https://open-box.readthedocs.io).

## Experimental Results

In this section, we compare Openbox with other popular hyperparameter optimization systems (or black-box optimization systems). We plot the ranks of tuning LightGBM on 25 datasets as follows:

<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/ranking_lgb_7.svg" style="zoom:30%;" />

We can have that, Openbox outperforms all the baselines on tuning LightGBM.

## Conclusion

In this article, we introduced how to tune LightGBM using OpenBox, and displayed its performance on hyperparameter optimization.

For more usages of OpenBox (e.g., multiple objectives, constraints, parallel running), please refer to our [documentation](https://open-box.readthedocs.io).

OpenBox is now open source on [Github](https://github.com/PKU-DAIR/open-box). We are actively accepting code contributions to the OpenBox project.

