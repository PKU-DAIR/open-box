# 多精度优化

在许多黑盒优化场景中，以全精度评估目标函数的代价是昂贵的（例如，训练一个深度神经网络的全部 epoch）。
**多精度优化** 利用更廉价的低精度评估（例如，更少的训练 epoch 或数据子集）来加速搜索，
只在最有希望的配置上花费完整资源。

OpenBox 通过 **调度器（scheduler）** 模块支持多精度优化，该模块实现了 Hyperband 风格的
逐次减半策略，包括 **BOHB** 和 **MFES** 变体。

## 快速示例

```python
from openbox import Optimizer, space as sp

cs = sp.Space()
cs.add_variable(sp.Real('learning_rate', 1e-5, 1e-1, log=True))
cs.add_variable(sp.Real('weight_decay', 1e-6, 1e-2, log=True))
cs.add_variable(sp.Int('batch_size', 16, 256))

def objective(config, resource_ratio=1.0):
    """
    目标函数必须接受 resource_ratio 参数。
    resource_ratio 范围从 0 到 1，其中 1.0 表示全精度。
    """
    epochs = int(100 * resource_ratio)
    val_loss = train_and_evaluate(config, epochs=max(epochs, 1))
    return {'objectives': [val_loss]}

optimizer = Optimizer(
    objective,
    cs,
    max_runs=50,
    advisor_type='default',
    surrogate_type='auto',
    scheduler_type='mfes',
    scheduler_kwargs={'R': 27, 'eta': 3},
)
history = optimizer.run()
```

## 核心概念

### 资源比例（Resource Ratio）

`resource_ratio` 是一个 (0, 1] 范围内的浮点数，用于控制精度级别：

- `resource_ratio=1.0` — 全精度评估（例如，100% 的训练 epoch）。
- `resource_ratio=0.11` — 低精度评估（例如，约 11% 的训练 epoch）。

使用多精度调度器时，目标函数 **必须** 接受 `resource_ratio` 作为关键字参数。
OpenBox 会在初始化时验证这一点。

### 逐次减半与 Bracket

多精度调度器使用 **逐次减半**：以低精度启动大量配置进行评估，保留前 1/η 的配置，
并将它们提升到下一个精度级别。这个过程重复进行，直到存活的配置以全精度进行评估。

一个 **bracket** 由参数 `s` 定义起始资源级别。更高的 `s` 意味着以更低的精度启动更多配置。
Bracket 在迭代中按 `s = s_max, s_max-1, ..., 0` 循环。

**R=27, η=3 的示例：**


| Bracket (s) | 阶段 0         | 阶段 1       | 阶段 2       | 阶段 3       |
| ----------- | ------------ | ---------- | ---------- | ---------- |
| s=3         | 27 配置 @ 1/27 | 9 配置 @ 1/9 | 3 配置 @ 1/3 | 1 配置 @ 1/1 |
| s=2         | 12 配置 @ 1/9  | 4 配置 @ 1/3 | 1 配置 @ 1/1 | —          |
| s=1         | 6 配置 @ 1/3   | 2 配置 @ 1/1 | —          | —          |
| s=0         | 4 配置 @ 1/1   | —          | —          | —          |


## 调度器类型


| 调度器          | 字符串              | 描述                                               |
| ------------ | ---------------- | ------------------------------------------------ |
| 全精度          | `'full'`         | 默认。始终以全精度评估（不使用多精度）。                             |
| BOHB         | `'bohb'`         | Hyperband 风格调度。仅全精度结果更新代理模型。使用标准 `Advisor`。      |
| MFES         | `'mfes'`         | 多精度调度。所有精度级别都更新 `MFAdvisor`，使 `mfgpe` 代理能够跨精度学习。 |
| Flatten      | `'flatten'`      | 扁平化的 BOHB bracket，用于更细粒度的全精度调度。                  |
| MFES Flatten | `'mfes_flatten'` | 结合扁平化调度与 MFES 历史管理。                              |


### `'bohb'` 与 `'mfes'` 的区别

BOHB 和 MFES 的关键区别在于如何使用低精度结果：

- **BOHB**：低精度评估仅用于淘汰。代理模型仅使用全精度数据训练。
使用标准 `Advisor`。
- **MFES**：所有精度级别通过 `MFAdvisor` 贡献给代理模型，`MFAdvisor` 为每个精度级别
维护独立的历史记录，并使用多精度代理（如 `mfgpe`）在精度间传递知识。

当 `scheduler_type='mfes'` 时，OpenBox 会自动：

1. 将 `advisor_type` 设置为 `'mf'`（创建 `MFAdvisor`）。
2. 如果未显式指定，将 `surrogate_type` 设置为 `'mfgpe'`。

## 调度器参数


| 参数    | 类型  | 默认值 | 描述                      |
| ----- | --- | --- | ----------------------- |
| `R`   | int | —   | 最大资源量（定义资源比例的分母）。       |
| `eta` | int | 3   | 淘汰比例因子。每个阶段保留前 1/η 的配置。 |


通过 `scheduler_kwargs` 传入：

```python
optimizer = Optimizer(
    objective,
    cs,
    max_runs=50,
    scheduler_type='bohb',
    scheduler_kwargs={'R': 9, 'eta': 3},
)
```

## 使用 Ask-and-Tell 接口

多精度优化也可以通过 `SMBO` 类使用，它是 `Optimizer` 的内部引擎：

```python
from openbox.optimizer.generic_smbo import SMBO

smbo = SMBO(
    objective_function=objective,
    config_space=cs,
    advisor_type='default',
    scheduler_type='mfes',
    scheduler_kwargs={'R': 27, 'eta': 3},
    max_runs=50,
)
history = smbo.run()
```

当调度器激活时，`SMBO.iterate()` 会在每次迭代中处理完整的逐次减半循环：请求一批配置，
以起始精度评估它们，淘汰表现最差的配置，提升存活者，重复直到最后阶段。

## 与搜索空间压缩结合

多精度调度与搜索空间压缩是 **正交的** 功能，可以同时使用。调度器控制评估精度，
压缩器变换搜索空间。

```python
optimizer = Optimizer(
    objective,
    cs,
    max_runs=50,
    scheduler_type='mfes',
    scheduler_kwargs={'R': 27, 'eta': 3},
    advisor_kwargs={
        'compressor_type': 'llamatune',
        'compressor_kwargs': {
            'adapter_alias': 'rembo',
            'le_low_dim': 5,
            'max_num_values': 50,
        },
    },
)
```

## MFAdvisor

`MFAdvisor` 扩展了 `Advisor` 以支持多精度工作流：

- 维护 `history_list` — 每个观测到的精度级别一个 `History`。
- 追踪 `resource_identifiers` — 到目前为止观测到的精度级别集合。
- `update_observation(observation, resource_ratio=1.0)`：
  - 始终将观测存储在精度特定的历史中。
  - 仅当 `resource_ratio == 1.0` 时更新主 `history`（用于标准 BO 循环）。
- 在生成建议之前，调用 `_prepare_mf_surrogate()` 更新多精度代理模型，
使用所有精度级别的数据。
