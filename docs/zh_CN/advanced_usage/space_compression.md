# 搜索空间压缩

在高维超参数优化中，搜索空间通常包含大量参数，但只有其中一小部分对目标函数有显著影响。
**搜索空间压缩** 通过一组压缩步骤组成的流水线来降低搜索空间的有效维度，从而使贝叶斯优化更加高效。

OpenBox 内置了一个压缩框架，该框架支持三类压缩：**维度选择**、**范围压缩** 和 **投影变换**。

## 快速示例

启用空间压缩最简单的方式是在创建 `Advisor` 或 `Optimizer` 时设置 `compressor_type`。

```python
from openbox import Advisor, space as sp

cs = sp.Space()
for i in range(20):
    cs.add_variable(sp.Real(f'x{i}', -10, 10))

advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    compressor_type='llamatune',
    compressor_kwargs={
        'adapter_alias': 'rembo',
        'le_low_dim': 5,
        'max_num_values': 50,
    },
)
```

## 压缩流水线

压缩被组织为有序 **步骤** 的流水线。每个步骤属于以下三类之一，并由一个短字符串标识。

### 维度选择步骤

根据参数重要性选择参数子集。

| 字符串 | 步骤类 | 描述 |
|--------|--------|------|
| `d_shap` | `SHAPDimensionStep` | 基于 SHAP 特征重要性的参数选择。支持迁移学习。 |
| `d_corr` | `CorrelationDimensionStep` | 基于 Spearman/Pearson 相关性的参数选择。支持迁移学习。 |
| `d_expert` | `ExpertDimensionStep` | 基于专家知识的参数选择。 |
| `d_adaptive` | `AdaptiveDimensionStep` | 在优化过程中自适应调整选择的参数数量。 |

### 范围压缩步骤

将参数取值范围缩小到高价值区域。

| 字符串 | 步骤类 | 描述 |
|--------|--------|------|
| `r_boundary` | `BoundaryRangeStep` | 基于均值 ± σ 的边界范围压缩。 |
| `r_shap` | `SHAPBoundaryRangeStep` | SHAP 加权的边界范围压缩。支持迁移学习。 |
| `r_kde` | `KDEBoundaryRangeStep` | 基于核密度估计的范围压缩。支持迁移学习。 |
| `r_expert` | `ExpertRangeStep` | 由专家指定的参数范围。 |

### 投影步骤

将参数空间变换到低维表示。

| 字符串 | 步骤类 | 描述 |
|--------|--------|------|
| `p_quant` | `QuantizationProjectionStep` | 对大范围整数参数进行量化。 |
| `p_rembo` | `REMBOProjectionStep` | 随机嵌入贝叶斯优化（REMBO）。 |
| `p_hesbo` | `HesBOProjectionStep` | 基于哈希的子空间贝叶斯优化（HesBO）。 |
| `p_kpca` | `KPCAProjectionStep` | 核 PCA 投影。 |

## 使用 `compressor_type` 快捷方式

OpenBox 提供了几种 `compressor_type` 快捷值，可自动构建压缩流水线。

### `'llamatune'` — 量化 + 投影

灵感来自 [LlamaTune](https://www.vldb.org/pvldb/vol15/p2953-kanellis.pdf) 方法。
组合量化与可选的 REMBO 或 HesBO 投影。

```python
advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    compressor_type='llamatune',
    compressor_kwargs={
        'adapter_alias': 'rembo',  # 'rembo'、'hesbo' 或 'none'
        'le_low_dim': 5,           # 低维目标维度
        'max_num_values': 50,      # 量化最大离散值数
    },
)
```

+ `adapter_alias`：投影方法。`'rembo'` 表示 REMBO，`'hesbo'` 表示 HesBO，
  `'none'` 表示仅量化。
+ `le_low_dim`：投影后的目标维度。
+ `max_num_values`：整数参数量化的最大离散值数。

### `'shap'` — SHAP 维度选择 + 边界范围压缩

使用 SHAP 重要性来选择参数，并可选择压缩其取值范围。
需要 `transfer_learning_history` 从历史数据中计算 SHAP 重要性。

```python
advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    transfer_learning_history=source_histories,
    compressor_type='shap',
    compressor_kwargs={
        'topk': 10,
        'top_ratio': 0.8,
        'sigma': 2.0,
    },
)
```

+ `topk`：保留的最重要参数数量。
+ `top_ratio`：用于边界估计的最佳配置比例。
+ `sigma`：边界宽度（均值 ± sigma × 标准差）。

### `'expert'` — 专家知识维度选择

选择由领域专家指定的参数。

```python
advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    compressor_type='expert',
    compressor_kwargs={
        'expert_params': ['x0', 'x3', 'x7'],
        'top_ratio': 0.9,
        'sigma': 2.0,
    },
)
```

## 构建自定义流水线

要实现完全控制，可使用 `compressor_type='pipeline'` 配合步骤字符串。

```python
advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    transfer_learning_history=source_histories,
    compressor_type='pipeline',
    compressor_kwargs={
        'step_strings': ['d_shap', 'r_boundary', 'p_rembo'],
        'step_params': {
            'd_shap': {'topk': 10},
            'r_boundary': {'top_ratio': 0.8, 'sigma': 2.0},
            'p_rembo': {'low_dim': 5, 'seed': 42},
        },
    },
)
```

也可以通过 `steps` 键传入预构建的步骤对象：

```python
from openbox.compressor import SHAPDimensionStep, BoundaryRangeStep

steps = [
    SHAPDimensionStep(strategy='shap', topk=10),
    BoundaryRangeStep(method='boundary', top_ratio=0.8, sigma=2.0),
]

advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    transfer_learning_history=source_histories,
    compressor_type='pipeline',
    compressor_kwargs={'steps': steps},
)
```

## 直接构造 Compressor 并传入

也可以直接构造 `Compressor` 实例，并通过 `compressor` 参数传给 `Advisor`。`SHAPDimensionStep`
与 `KDEBoundaryRangeStep` 需要历史数据来计算重要性与范围，请像使用 `'shap'` 快捷方式一样传入
`transfer_learning_history`。

```python
from openbox.compressor import Compressor, SHAPDimensionStep, KDEBoundaryRangeStep

steps = [
    SHAPDimensionStep(strategy='shap', topk=10),
    KDEBoundaryRangeStep(top_ratio=0.8, kde_coverage=0.6),
]
compressor = Compressor(config_space=cs, steps=steps)

advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    transfer_learning_history=source_histories,
    compressor=compressor,
)
```

## 空间概念

当搜索空间压缩激活时，OpenBox 会使用多个空间：

+ **原始空间** — 用户定义的完整配置空间。
+ **采样空间** — 采集函数优化器用于提出新配置的空间。
+ **代理空间** — 用于代理模型训练和预测的空间，是流水线的最终输出。
+ **反投影空间** — 第一个投影步骤之前的空间，用于将投影后的配置映射回原始空间进行评估。

```
原始空间
    ↓ [维度选择]
降维空间
    ↓ [范围压缩]
范围压缩空间
    ↓ [投影]
压缩空间
    ├── 采样空间：用于生成新配置
    └── 代理空间：用于模型训练
```

## 自适应压缩

`AdaptiveDimensionStep` 可以根据更新策略在优化过程中动态调整所选参数的数量。

```python
advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    compressor_type='pipeline',
    compressor_kwargs={
        'step_strings': ['d_adaptive'],
        'step_params': {
            'd_adaptive': {
                'importance_calculator': 'shap',
                'update_strategy': 'periodic',
                'update_strategy_kwargs': {'period': 10},
                'initial_topk': 15,
                'reduction_ratio': 0.2,
                'min_dimensions': 5,
            },
        },
    },
)
```

在优化循环中触发自适应更新：

```python
for i in range(max_iter):
    config = advisor.get_suggestion()
    result = objective(config)
    observation = Observation(config=config, objectives=result['objectives'])
    advisor.update_observation(observation)

    updated = advisor.update_compression(advisor.history)
    if updated:
        print(f'第 {i} 次迭代时压缩策略已更新')
```

### 更新策略

| 策略 | 行为 |
|------|------|
| `'periodic'` | 每 N 次迭代更新一次，减少维度。 |
| `'stagnation'` | 优化停滞时增加维度。 |
| `'improvement'` | 连续改善后减少维度。 |
| `'hybrid'` | 组合周期性、停滞和改善策略。 |

## 与迁移学习的集成

空间压缩与迁移学习自然配合。当提供 `transfer_learning_history` 时，压缩器会使用源任务数据来：
1. 计算参数重要性（用于 SHAP/相关性维度选择）。
2. 估计取值范围（用于边界/KDE 范围压缩）。
3. 将源任务历史转换到压缩空间，供代理模型使用。

```python
advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    transfer_learning_history=source_histories,
    surrogate_type='tlbo_rgpe_gp',
    compressor_type='shap',
    compressor_kwargs={'topk': 10, 'top_ratio': 0.8},
)
```
