# v0.8.2 - Mar 5, 2024

### Major Changes
+ Update pytest (#81).
+ Update early stopping algorithm.
+ Enhance documentation.
+ Drop support for Python 3.7 (end of life on 2023-06-27).

### Bug Fixes
+ Fix bug in distributed optimizer on Windows (#61).
+ Fix bug in async parallel optimizer.
+ Fix several bugs.

### API Changes
+ Update API for AcquisitionFunctionMaximizer (#75).


# v0.8.1 - Mar 8, 2023

### Bug Fixes
+ Fix several bugs in transfer learning.
+ Fix several bugs & improve usability of HTMLVisualizer (#58).
+ Update time limit function to support multi-platform and multi-version of python. Update API. (#59)

### API Changes
+ Optimizer: `runtime_limit` -> `max_runtime`. `time_limit_per_trial` -> `max_runtime_per_trial` (#59).

### Minor Changes
+ Update html theme for docs.


# v0.8.0 - Dec 18, 2022

## Highlights
+ Add HTML visualization for the optimization process (#48).
  + Provide basic charts for objectives and constraints.
  + Provide advanced functions, including surrogate fitting analysis and hyperparameter importance analysis.
+ Update transfer learning (#54).
  + API change: for transfer learning data, user should provide a `List[History]` as `transfer_learning_history`,
    instead of a `OrderedDict[config, perf]` as `history_bo_data` (#54, 4641d7cf).
  + Examples and docs are updated.
+ Refactor History object (0bce5800).
  + Rename `HistoryContainer` to `History`.
  + Simplify data structure and provide convenient APIs.
  + Rewrite all methods, including data obtaining, plotting, saving/loading, etc.

### Backwards Incompatible Changes
+ API change: `objs` are renamed to `objectives`. `num_objs` are renamed to `num_objectives` (ecd5928a).
+ Change objective value of failed trials from MAXINT to np.inf (da88bd24).
+ Drop support for Python 3.6 (end of life on Dec 23, 2021).

### Other Changes
+ Add BlendSearch, LineBO and SafeOpt (experimental) (#40).
+ Add color logger. Provide fine-grained control of logging options (e.g., log level).
+ Rewrite python packaging of the project (#55).
+ Update Markdown parser in docs to myst-parser. recommonmark is deprecated.
+ Add pytest for examples.
+ Use GitHub Actions for CI/CD.

### Bug Fixes
+ Fix error return type of generic advisor and update sampler (Thanks @yezoli) (#44).
+ Consider constraints in plot_convergence (#47).


# v0.7.18 - Nov 14, 2022

+ Add ConditionedSpace to support complex conditions between hyperparameters (#37).
+ Numerous bug fixes.


# v0.0.1 - May 2, 2019

Project started.
