# 服务简介
 
**OpenBox** 的设计原则是把BBO作为服务提供给用户。

<img src="../../imgs/sys_framework.svg" width="90%" class="align-center">

<center>OpenBox的体系结构。</center>

**OpenBox** 系统的体系结构包含以下五个组成构件：

+ **Service Master** 负责节点管理，负载均衡以及容错。

+ **Task Database** 保存所有任务的历史信息和历史状态。

+ **Suggestion Server** 给每个任务生成新的配置。

+ **REST API** 使用RESTful APIs来连接Workers和Suggestion Service。

+ **Evaluation workers** 由用户拥有和定义任务。


## 并行基础设施

**OpenBox**可以同时为大量任务生成配置。仅仅使用单台机器不足以处理工作负载。
因此，我们的建议将服务部署在多台机器上，每个机器是一个**Suggestion Server**。
每个**Suggestion Server**并行对多个任务生成建议。
这为我们提供了一个可大规模扩展的建议基础设施。
另一个主要组件是**Service Master**，它负责管理**Suggestion Server**，并平衡工作负载。
它作为统一的端点，接受工人的请求；
这样，每个工人就不需要知道调度细节。
工作者从**Suggestion Server**请求新配置，**Suggestion Server**根据自动算法选择模块确定的算法生成这些配置。
具体来说，在这个过程中，**Suggestion Server**利用基于局部惩罚的并行化机制和迁移学习框架来提高采样效率。


一个主要的设计考虑是维护一个容许误差的生产系统，因为机器崩溃是不可避免的。
在**OpenBox**中，**Service Master**监视每个服务器的状态，并保留一个活动服务器表。
当新任务到来时，**Service Master**会将其分配给活动服务器并记录此绑定信息。
如果一台服务器关闭，它的任务将被主服务器调度到一个新的服务器，以及存储在**Task Database**中的相关优化历史记录。
负载平衡是进行此类任务分配的最重要准则之一。
另外，**Service Master**的快照存储在远程数据库服务中；如果主机关闭，我们可以通过重新启动节点并从数据库中获取快照来恢复它。


## 服务接口

### 任务描述语言
为了便于使用，我们设计了一个任务描述语言（TDL）来定义优化任务。
TDL的核心部分是定义搜索空间，包括每个参数的类型、取值范围以及它们之间的关系。
**OpenBox**支持FLOAT、INTEGER、ORDINAL和CATEGORICAL等参数类型。
此外，用户还可以对参数添加条件，进一步限制搜索空间。用户还可以在TDL中指定时间预算、任务类型、工作结点数量、并行策略和使用历史。



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

这里有一个TDL的例子。
它定义了四个不同类型的参数*x1-4*和一个条件*cdn1*，表示只有当*x3 = "a3"*时，*x1*才处于激活状态。
时间预算为3小时，并行策略为异步，并启用了迁移学习。




### 基本工作流

给定任务的TDL，**OpenBox**的基本工作流实现如下：

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

这里**Evaluate**是用户提供的目标函数的评估过程。
通过调用**CreateTask**，worker获得一个全局唯一标识符**global_task_id**。
所有注册了相同**global_task_id**的工作人员都保证链接到同一个任务，从而支持并行评估。
当任务未完成时，工作者继续调用**GetSuggestions**和**UpdateObservations**从建议服务中提取建议并更新其相应的观察结果。

### 接口

用户可以通过如下接口和 **OpenBox** 进行交互：

+ **Register**: 它接受global_task_id作为输入，该id是在从workers调用CreateTask时创建的，并将当前worker与相应的任务绑定。
  这允许在多个worker之间共享优化过程的历史记录。

+ **Suggest**: 根据对当前任务的历史观察，它提出了下一步要评估的配置。

+ **Update**: 这种方法用从worker那里获得的观察更新优化历史。观察结果包括三个部分：目标值、约束结果和评价信息。

+ **StopEarly**: 它返回一个布尔值，指示是否应提前停止当前计算。

+ **Extrapolate**: 它用延拓的方法，并以交互方式向用户提供资源感知建议。

