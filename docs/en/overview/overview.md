# Overview
Black–box optimization (BBO) is the task of optimizing an objective function within a limited budget for function evaluations.
''Black-box'' means that the objective function has no analytical form so that information such as the derivative of the objective function is unavailable.
Since the evaluation of objective functions is often expensive, the goal of black-box optimization is to find a configuration that approaches the global optimum as rapidly as possible.

Traditional single-objective BBO has many applications, including:

+ Automatic A/B testing.
+ Experimental design.
+ Database knob tuning.
+ **Automatic hyperparameter tuning.**

Recently, generalized BBO emerges and has been applied to many areas:

+ Processor architecture and circuit design.
+ Resource allocation.
+ Automatic chemical design.

Generalized BBO requires more general functionalities that may not be supported by traditional BBO,
such as multiple objectives and constraints.

## Design Principle

OpenBox is an efficient system designed for generalized Black-box Optimization (BBO).
Its design satisfies the following desiderata:

+ **Ease of use**: Minimal user effort, and user-friendly visualization for tracking and managing BBO tasks.

+ **Consistent performance**: Host state-of-the-art optimization algorithms; Choose the proper algorithm automatically.

+ **Resource-aware management**: Give cost-model-based advice to users, e.g., minimal workers or time-budget.

+ **Scalability**: Scale to dimensions on the number of input variables, objectives, tasks, trials, and parallel evaluations.

+ **High efficiency**: Effective use of parallel resources, system optimization with transfer-learning and multi-fidelities, etc.

+ **Fault tolerance**, **extensibility**, and **data privacy protection**.


The figure below shows the high-level architecture of OpenBox service.

<p align="center">
<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/sys_framework.svg" width="90%">
</p>

## Main Components

+ **Service Master** is responsible for node management, load balance, and fault tolerance.

+ **Task Database** holds the history and states of all tasks. 

+ **Suggestion Server** generates new configurations for each task.

+ **REST API** connects users/workers and suggestion service via RESTful APIs. 

+ **Evaluation workers** are provided and owned by the users.


## Deployment Artifacts

### Standalone Python package
Like other open-source packages, OpenBox has a frequent release cycle. Users can install the package via Pypi or
source code on [GitHub](https://github.com/PKU-DAIR/open-box). For more installation details, refer to [Installation Guide](../installation/installation_guide.md).

### Distributed BBO service
We adopt the "BBO as a service" paradigm and implement OpenBox as a managed general service for black-box optimization.
Users can access this service via RESTful API conveniently, regardless of other issues such as
environment setups, software maintenance, and execution optimization. Moreover, OpenBox also provide
Web UI for users to track and manage their running tasks. For deployment details, refer to [Deployment Guide](../openbox_as_service/service_deployment.md).


## Performance Comparison
We compare OpenBox with six competitive open-source BBO systems on tuning LightGBM using 25 datasets. 
The performance rank (the lower, the better) is shown in the following figure. 
For dataset information and more experimental results, please refer to our [published article](https://arxiv.org/abs/2106.00421).

<p align="center">
<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/ranking_lgb_7.svg" width="80%">
</p>

