# Introduction of OpenBox as Service

The design of **OpenBox** follows the paradigm of providing “BBO as a service”.

<p align="center">
<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/sys_framework.svg" width="90%">
</p>

<center>Architecture of OpenBox.</center>

The system architecture of **OpenBox** includes five main components:

+ **Service Master** is responsible for node management, load balance, and fault tolerance.

+ **Task Database** holds the history and states of all tasks. 

+ **Suggestion Server** generates new configurations for each task.

+ **REST API** connects users/workers and suggestion service via RESTful APIs. 

+ **Evaluation workers** are provided and owned by the users.


## Parallel Infrastructure

**OpenBox** is designed to generate suggestions for a large number of
tasks concurrently, and a single machine would be insufficient to
handle the workload. Our suggestion service is therefore deployed
across several machines, called **suggestion servers**. Each **suggestion
server** generates suggestions for several tasks in parallel, giving
us a massively scalable suggestion infrastructure. Another main
component is **service master**, which is responsible for managing
the **suggestion servers** and balancing the workload. It serves as the
unified endpoint, and accepts the requests from workers; in this
way, each worker does not need to know the dispatching details.
The worker requests new configurations from the **suggestion server**
and the **suggestion server** generates these configurations based on an
algorithm determined by the automatic algorithm selection module.
Concretely, in this process, the **suggestion server** utilizes the local
penalization based parallelization mechanism and transfer learning
framework to improve the sample efficiency.

One main design consideration is to maintain a fault-tolerant production
system, as machine crash happens inevitably. In **OpenBox**,
the **service master** monitors the status of each server and preserves
a table of active servers. When a new task comes, the **service master**
will assign it to an active server and record this binding information.
If one server is down, its tasks will be dispatched to a new server by
the master, along with the related optimization history stored in the
**task database**. Load balance is one of the most important guidelines
to make such task assignments. In addition, the snapshot of **service
master** is stored in the remote database service; if the master is
down, we can recover it by restarting the node and fetching the
snapshot from the database.

## Service Interfaces

### Task Description Language
For ease of usage, we design a
Task Description Language (TDL) to define the optimization task.
The essential part of TDL is to define the search space, which includes
the type and bound for each parameter and the relationships
among them. The parameter types — FLOAT, INTEGER, ORDINAL
and CATEGORICAL are supported in **OpenBox**. In addition, users
can add conditions of the parameters to further restrict the search
space. Users can also specify the time budget, task type, number of
workers, parallel strategy and use of history in TDL. 

```
task_config = {
    "parameter": {
        "x1": {"type": "float", "default": 0,
            "bound": [-5, 10]} ,
        "x2": {"type": "int", "bound": [0, 15]} ,
        "x3": {"type": "cat", "default": "a1",
            "choice": ["a1", "a2", "a3"]} ,
        "x4": {"type": "ord", "default": 1,
            "choice": [1, 2, 3]}} ,
    "condition": {
        "cdn1": {"type": "equal", "parent": "x3",
            "child": "x1", "value": "a3"}} ,
    "number_of_trials": 200 ,
    "time_budget": 10800 ,
    "task_type": "soc",
    "parallel_strategy": "async",
    "worker_num": 10,
    "use_history": True
    }
```

Here's an example of TDL. It defines four parameters *x1-4* of different
types and a condition *cdn1*, which indicates that *x1* is active only
if *x3 = "a3"*. The time budget is three hours, the parallel strategy
is *async*, and transfer learning is enabled.


### BasicWorkflow

Given the TDL for a task, the basic workflow of **OpenBox** is implemented as follows:

```python
# Register the worker with a task .
global_task_id = worker.CreateTask(task_tdl)
worker.BindTask(global_task_id)
while not worker.TaskFinished():
    # Obtain a configuration to evaluate.
    config = worker.GetSuggestions()
    # Evaluate the objective function.
    result = Evaluate(config)
    # Report the evaluated results to the server.
    worker.UpdateObservations(config, result)
```

Here **Evaluate** is the evaluation procedure of objective function
provided by users. By calling **CreateTask**, the worker obtains a
globally unique identifier **global_task_id**. All workers registered
with the same **global_task_id** are guaranteed to link with the
same task, which enables parallel evaluations. While the task is
not finished, the worker continues to call **GetSuggestions** and
**UpdateObservations** to pull suggestions from the suggestion
service and update their corresponding observations.

### Interfaces

Users can interact with the **OpenBox** service via
a **REST API**. We list the most important service calls as follows:

+ **Register**: It takes as input the global_task_id, which is
created when calling CreateTask from workers, and binds
the current worker with the corresponding task. This allows
for sharing the optimization history across multiple workers.

+ **Suggest**: It suggests the next configurations to evaluate,
given the historical observations of the current task.

+ **Update**: This method updates the optimization history with
the observations obtained from workers. The observations
include three parts: the values of the objectives, the results
of constraints, and the evaluation information.

+ **StopEarly**: It returns a boolean value that indicates whether
the current evaluation should be stopped early.

+ **Extrapolate**: It uses performance-resource extrapolation,
and interactively gives resource-aware advice to users.

