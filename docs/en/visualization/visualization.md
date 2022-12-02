# HTML Visualization

<font color=#FF0000>(New Feature!)</font> **OpenBox** provides HTML visualization 
for user to monitor and analyze the optimization process.
In this tutorial, we will explain the usage of HTML visualization in **OpenBox**.

## Enable HTML Visualization

We assume that you already know how to set up a problem in **OpenBox**. 
If not, please refer to the [Quick Start Tutorial](../quick_start/quick_start).

Here we use an example problem from 
[Multi-Objective Black-box Optimization with Constraints](../examples/multi_objective_with_constraint).

To enable HTML visualization, just set `visualization` = `basic` or `advanced` in `Optimizer`:

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

There are 3 options for `visualization`:
+ **'none'**: Run the task without visualization. 
No additional files are generated. Better for running massive experiments.
+ **'basic'**: Run the task with basic visualization, 
including visualization plots for objectives and constraints.
+ **'advanced'**: Enable the visualization with advanced functions, 
including model fitting analysis and hyperparameter importance analysis.

**<font color=#FF0000>Note:</font>** to execute the hyperparameter importance analysis, additional packages, 
`shap` and `lightgbm`, are required to be installed (`pip install shap lightgbm`).

Once the `Optimizer` is initialized, an HTML page will be generated in `${logging_dir}/history/${task_id}/`.
Open the HTML page in your browser, you can see the visualization of the optimization process.

During the optimization, click the `Refresh` button to update the visualization result.

### Enable HTML Visualization after the Optimization

If you forget to set `visualization` in `Optimizer`, don't worry,
you can also view the visualization result after the optimization is finished:
```python
history = opt.get_history()
history.visualize_html(
    show_importance=True,
    verify_surrogate=True,
    optimizer=opt,
)
```

An HTML page is then generated in `${logging_dir}/history/${task_id}/`.

Also note that if `show_importance=True`, additional packages, `shap` and `lightgbm`, 
are required to be installed (`pip install shap lightgbm`).


## Basic Visualization

### 1 Objective Function

#### 1.1 Objective Value Chart

This shows the objective value of every iteration. 

For **constrained problems**, 
observations that meet the constriant will be shown as circle <font color=#0000FF>$\bigcirc$</font>,
Otherwise, triangle <font color=#0000FF>$\triangle$</font>.

<img src="../../imgs/visualization/obj_value.png" width="80%" class="align-center">

<br>

#### 1.2 Constraint Value Chart

This visualization is only for **constrained problems**.

This shows the constraint value of every iteration. By default, Non-positive constraint values (**”<=0”**) imply feasibility.

<img src="../../imgs/visualization/cons_value.png" width="80%" class="align-center">

<br>

#### 1.3 Parallel Coordinates Plot

This shows the values of parameters and objective values of individual observation each round.

<img src="../../imgs/visualization/parallel.png" width="80%" class="align-center">

<br>

### 2 Pareto
This part is only available for **muti-objective problems**.

In multi-objective problems, since we don't know which objective is the most important, we find a set of pareto optimal solutions. A pareto optimal solution means that it cannot be improved in any of the objectives without degrading at least one of the other objective. All pareto optimal solutions form a pareto frontier. Our aim is to maximize the hypervolume from a worst solution to the pareto frontier.

#### 2.1 Pareto Frontier

Visualization of pareto frontier is only available for **two or three objectives problems**.

Pareto frontier will be shown as a curve (2-obj) or a surface (3-obj). 
For **constrained problems**, 
observations that meet the constriant will be shown as circle <font color=#0000FF>$\bigcirc$</font>. 
Otherwise, triangle <font color=#0000FF>$\triangle$</font>.

<img src="../../imgs/visualization/pareto_front.png" width="80%" class="align-center">

<br>

#### 2.2 Pareto Frontier Hypervolume

This shows the hypervolume surrounded by the pareto frontier in each iteration.

<img src="../../imgs/visualization/pareto_hypervolume.png" width="80%" class="align-center">

<br>

### 3 Historical Configurations

This table records data of every run of optimization. If the data is too long to show all, you can click the **"..."** beside it to see the whole data.

<img src="../../imgs/visualization/history.png" width="80%" class="align-center">

<br>

## Advanced Visualization

### 1 Surrogate Model

A surrogate model is trained to approximate the predictions of a black box model. We visualize surrogate model to see its performance.

#### 1.1 Predicted Objective Value

X-axis is the predicted objective value from surrogate model. Y-axis is the true value. The closer the dot is to the line y=x, the better the prediction of surrogate model is.

<img src="../../imgs/visualization/surrogate_obj.png" width="80%" class="align-center">

<br>

#### 1.2 Predicted Objective Value Rank

This chart is similar to *Predicted Objective Value*. But here we predict the rank of a few objective value. X-axis is the predicted objective value rank from surrogate model. Y-axis is the true value rank. The closer the dot is to the line y=x, the better the prediction of surrogate model is.

<img src="../../imgs/visualization/surrogate_obj_rank.png" width="80%" class="align-center">

<br>

#### 1.3 Predicted Constraint Value

This chart is only for **constrained problems**.

Besides objective value, we can also use surrogate model to predict constraint value. This chart is similar to *Predicted Objective Value*, except that we predict constraint value here.

<img src="../../imgs/visualization/surrogate_cons.png" width="80%" class="align-center">

<br>

### 2 Parameter Importance

We use **SHAP** (SHapley Additive exPlanations) approach to evaluate parameter importance. More information about SHAP, please see [**SHAP documentation**](https://shap.readthedocs.io/en/latest/).

#### 2.1 Overall Parameter Importance

This chart shows importance of each parameter to the objective. The higher the importance value is, the greater this parameter influences the objective, whether in a positive way or an negative way.

<img src="../../imgs/visualization/importance_obj.png" width="80%" class="align-center">

<br>

#### 2.2 Overall Parameter Importance of Constraints

This chart shows the importance of each parameter to the constraints. The higher the importance value is, the greater this parameter influences the constraint, whether in a positive way or an negative way.the objective, whether in a positive way or an negative way.

<img src="../../imgs/visualization/importance_cons.png" width="80%" class="align-center">

<br>

#### 2.3 Single Parameter Importance of Objectives

This chart shows how the objective depends on the given parameter. X-axis is the value of the parameter. Y-axis is the SHAP value of it. The absolute value of SHAP value represents the influence intensity. Positive value means a positive correlation. Negative value means a negative correlation. You can click the label on the top to switch parameter.

For **multi-objective problems**, You can select the objective from the drop-down box above the figure.

<img src="../../imgs/visualization/single_obj.png" width="80%" class="align-center">

<br>

#### 2.4 Single Parameter Importance of Constraints

This chart shows how the constraints depends on the given parameter. X-axis is the value of the parameter. Y-axis is the SHAP value of it. The absolute value of SHAP value represents the influence intensity. Positive value means a positive correlation. Negative value means a negative correlation. You can click the label on the top to switch parameter.

You can select the constraint from the drop-down box above the figure if there are **more than one constraints**.

<img src="../../imgs/visualization/single_cons.png" width="80%" class="align-center">

<br>
