# Multi-Fidelity Optimization

In many black-box optimization scenarios, evaluating the objective function at full fidelity
is expensive (e.g., training a deep neural network for all epochs). **Multi-fidelity
optimization** exploits cheaper, lower-fidelity evaluations (e.g., fewer training epochs or
a data subset) to accelerate the search, only spending full resources on the most promising
configurations.

OpenBox supports multi-fidelity optimization through a **scheduler** module that implements
Hyperband-style successive halving strategies, including **BOHB** and **MFES** variants.

## Quick Example

```python
from openbox import Optimizer, space as sp

cs = sp.Space()
cs.add_variable(sp.Real('learning_rate', 1e-5, 1e-1, log=True))
cs.add_variable(sp.Real('weight_decay', 1e-6, 1e-2, log=True))
cs.add_variable(sp.Int('batch_size', 16, 256))

def objective(config, resource_ratio=1.0):
    """
    The objective function must accept a `resource_ratio` parameter.
    resource_ratio ranges from 0 to 1, where 1.0 means full fidelity.
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

## Key Concepts

### Resource Ratio

The `resource_ratio` is a float in (0, 1] that controls the fidelity level:
+ `resource_ratio=1.0` — full fidelity evaluation (e.g., 100% of training epochs).
+ `resource_ratio=0.11` — low fidelity (e.g., ~11% of training epochs).

Your objective function **must** accept `resource_ratio` as a keyword argument when using
multi-fidelity schedulers. OpenBox validates this at initialization.

### Successive Halving and Brackets

Multi-fidelity schedulers use **successive halving**: start many configurations at low
fidelity, evaluate them, keep the top 1/η fraction, and promote them to the next fidelity
level. This process repeats until the surviving configurations are evaluated at full fidelity.

A **bracket** with parameter `s` defines the initial resource level. Higher `s` means
starting with more configurations at lower fidelity. Brackets cycle through
`s = s_max, s_max-1, ..., 0` across iterations.

**Example with R=27, η=3:**

| Bracket (s) | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|:-----------:|:-------:|:-------:|:-------:|:-------:|
| s=3 | 27 configs @ 1/27 | 9 configs @ 1/9 | 3 configs @ 1/3 | 1 config @ 1/1 |
| s=2 | 12 configs @ 1/9 | 4 configs @ 1/3 | 1 config @ 1/1 | — |
| s=1 | 6 configs @ 1/3 | 2 configs @ 1/1 | — | — |
| s=0 | 4 configs @ 1/1 | — | — | — |

## Scheduler Types

| Scheduler | String | Description |
|-----------|--------|-------------|
| Full Fidelity | `'full'` | Default. Always evaluates at full fidelity (no multi-fidelity). |
| BOHB | `'bohb'` | Hyperband-style scheduling. Only full-fidelity results update the surrogate model. Uses a standard `Advisor`. |
| MFES | `'mfes'` | Multi-fidelity scheduling. All fidelity levels update the `MFAdvisor`, enabling the `mfgpe` surrogate to learn across fidelities. |
| Flatten | `'flatten'` | Flattened BOHB brackets for finer-grained full-fidelity scheduling. |
| MFES Flatten | `'mfes_flatten'` | Combines flatten scheduling with MFES history management. |

### `'bohb'` vs `'mfes'`

The key difference between BOHB and MFES lies in how low-fidelity results are used:

+ **BOHB**: Low-fidelity evaluations are used only for elimination (successive halving).
  The surrogate model is trained only on full-fidelity data. Uses a standard `Advisor`.
+ **MFES**: All fidelity levels contribute to the surrogate model through `MFAdvisor`,
  which maintains separate history lists per fidelity level and uses a multi-fidelity
  surrogate (e.g., `mfgpe`) to transfer knowledge across fidelities.

When `scheduler_type='mfes'`, OpenBox automatically:
1. Sets `advisor_type` to `'mf'` (creating an `MFAdvisor`).
2. Sets `surrogate_type` to `'mfgpe'` if not explicitly specified.

## Scheduler Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `R` | int | — | Maximum resource amount (denominator for resource ratios). |
| `eta` | int | 3 | Reduction factor. Top 1/η configurations survive each stage. |

These are passed via `scheduler_kwargs`:

```python
optimizer = Optimizer(
    objective,
    cs,
    max_runs=50,
    scheduler_type='bohb',
    scheduler_kwargs={'R': 9, 'eta': 3},
)
```

## Using the Ask-and-Tell Interface

Multi-fidelity optimization is also supported through the `SMBO` class, which is the
internal engine of `Optimizer`:

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

When the scheduler is active, `SMBO.iterate()` handles the full successive halving loop
within each iteration: requesting a batch of configurations, evaluating them at the starting
fidelity level, eliminating the worst performers, promoting survivors, and repeating until
the final stage.

## Combining with Space Compression

Multi-fidelity scheduling and space compression are **orthogonal** features that can be
used together. The scheduler controls evaluation fidelity, while the compressor transforms
the search space.

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

The `MFAdvisor` extends `Advisor` to support multi-fidelity workflows:

+ Maintains a `history_list` — one `History` per observed fidelity level.
+ Tracks `resource_identifiers` — the set of fidelity levels seen so far.
+ `update_observation(observation, resource_ratio=1.0)`:
  - Always stores the observation in the fidelity-specific history.
  - Only updates the main `history` (used by the standard BO loop) when `resource_ratio == 1.0`.
+ Before generating suggestions, calls `_prepare_mf_surrogate()` which updates the
  multi-fidelity surrogate model with data from all fidelity levels.
