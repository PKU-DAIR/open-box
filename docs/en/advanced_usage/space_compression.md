# Space Compression

In high-dimensional hyperparameter optimization, the search space often contains many parameters,
but only a subset of them significantly influences the objective function.
**Space compression** reduces the effective dimensionality of the search space through a
pipeline of compression steps, making Bayesian optimization more efficient.

OpenBox provides a built-in compression framework that supports three categories of
compression: **dimension selection**, **range compression**, and **projection**.

## Quick Example

The simplest way to enable space compression is to set `compressor_type` when creating an
`Advisor` or `Optimizer`.

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

## Compression Pipeline

Compression is organized as a **pipeline** of ordered steps. Each step belongs to one of three
categories and is identified by a short string.

### Dimension Selection Steps

Select a subset of parameters by importance.

| String | Step Class | Description |
|--------|-----------|-------------|
| `d_shap` | `SHAPDimensionStep` | SHAP-based feature importance selection. Supports transfer learning. |
| `d_corr` | `CorrelationDimensionStep` | Spearman/Pearson correlation-based selection. Supports transfer learning. |
| `d_expert` | `ExpertDimensionStep` | Expert-specified parameter selection. |
| `d_adaptive` | `AdaptiveDimensionStep` | Adaptively adjusts the number of selected parameters during optimization. |

### Range Compression Steps

Narrow parameter value ranges to high-value regions.

| String | Step Class | Description |
|--------|-----------|-------------|
| `r_boundary` | `BoundaryRangeStep` | Mean ± σ boundary-based compression. |
| `r_shap` | `SHAPBoundaryRangeStep` | SHAP-weighted boundary compression. Supports transfer learning. |
| `r_kde` | `KDEBoundaryRangeStep` | KDE-based range compression. Supports transfer learning. |
| `r_expert` | `ExpertRangeStep` | Expert-specified parameter ranges. |

### Projection Steps

Transform the parameter space into a lower-dimensional representation.

| String | Step Class | Description |
|--------|-----------|-------------|
| `p_quant` | `QuantizationProjectionStep` | Integer quantization for large-range integer parameters. |
| `p_rembo` | `REMBOProjectionStep` | Random Embedding Bayesian Optimization. |
| `p_hesbo` | `HesBOProjectionStep` | Hashing-Enhanced Subspace Bayesian Optimization. |
| `p_kpca` | `KPCAProjectionStep` | Kernel PCA projection. |

## Using `compressor_type` Shortcuts

OpenBox provides several shortcut `compressor_type` values that automatically construct a
pipeline.

### `'llamatune'` — Quantization + Projection

Inspired by the [LlamaTune](https://www.vldb.org/pvldb/vol15/p2953-kanellis.pdf) method.
Combines quantization with optional REMBO or HesBO projection.

```python
advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    compressor_type='llamatune',
    compressor_kwargs={
        'adapter_alias': 'rembo',  # 'rembo', 'hesbo', or 'none'
        'le_low_dim': 5,           # low-dimensional target dimension
        'max_num_values': 50,      # max discrete values for quantization
    },
)
```

+ `adapter_alias`: Projection method. `'rembo'` for REMBO, `'hesbo'` for HesBO,
  or `'none'` for quantization only.
+ `le_low_dim`: Target dimensionality of the projected space.
+ `max_num_values`: Maximum number of discrete values for integer parameter quantization.

### `'shap'` — SHAP Dimension Selection + Boundary Range

Uses SHAP-based importance to select parameters and optionally compresses their ranges.
Requires `transfer_learning_history` to compute SHAP importances from historical data.

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

+ `topk`: Number of top important parameters to keep.
+ `top_ratio`: Fraction of best configurations used for boundary estimation.
+ `sigma`: Width of the boundary (mean ± sigma × std).

### `'expert'` — Expert Knowledge Dimension Selection

Selects parameters specified by domain experts.

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

## Building a Custom Pipeline

For full control, use `compressor_type='pipeline'` with step strings.

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

You can also pass pre-built step objects via the `steps` key:

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

## Using a Pre-built Compressor

You can also construct a `Compressor` instance directly and pass it to the `Advisor` via the
`compressor` argument. Steps such as `SHAPDimensionStep` and `KDEBoundaryRangeStep` need source
histories for importance and range estimation—pass `transfer_learning_history` as you would for
the `'shap'` shortcut.

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

## Space Concepts

When space compression is active, OpenBox works with multiple spaces:

+ **Original space** — the full configuration space as defined by the user.
+ **Sample space** — the space used by the acquisition optimizer to propose new configurations.
+ **Surrogate space** — the space used for surrogate model training and prediction; the final
  output of the pipeline.
+ **Unprojected space** — the space before the first projection step, used to map projected
  configurations back to the original space for evaluation.

```
Original space
    ↓ [Dimension Selection]
Dimension-reduced space
    ↓ [Range Compression]
Range-compressed space
    ↓ [Projection]
Compressed space
    ├── Sample space: for generating new configurations
    └── Surrogate space: for model training
```

## Adaptive Compression

The `AdaptiveDimensionStep` can dynamically adjust the number of selected parameters during
optimization based on an update strategy.

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

To trigger adaptive updates during the optimization loop:

```python
for i in range(max_iter):
    config = advisor.get_suggestion()
    result = objective(config)
    observation = Observation(config=config, objectives=result['objectives'])
    advisor.update_observation(observation)

    updated = advisor.update_compression(advisor.history)
    if updated:
        print(f'Compression policy updated at iteration {i}')
```

### Update Strategies

| Strategy | Behavior |
|----------|----------|
| `'periodic'` | Update every N iterations. Reduce dimensions. |
| `'stagnation'` | Increase dimensions when optimization stagnates. |
| `'improvement'` | Reduce dimensions after consecutive improvements. |
| `'hybrid'` | Combines periodic, stagnation, and improvement strategies. |

## Integration with Transfer Learning

Space compression works naturally with transfer learning. When `transfer_learning_history`
is provided, the compressor uses the source task data to:
1. Compute parameter importances (for SHAP/Correlation-based dimension selection).
2. Estimate value ranges (for boundary/KDE range compression).
3. Transform source task histories to the compressed space for the surrogate model.

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
