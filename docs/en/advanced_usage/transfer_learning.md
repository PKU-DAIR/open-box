# Transfer Learning

When performing BBO, users often run tasks that are similar to
previous ones. This observation can be used to speed up the current task.
Compared with Vizier, which only provides limited transfer learning
functionality for single-objective BBO problems, OpenBox employs
a general transfer learning framework with the following
advantages:

1) Support for generalized black-box optimization problems;

2) Compatibility with most Bayesian optimization methods.

OpenBox takes as input observations from $K + 1$ tasks: $D^1$, ...,
$D^K$ for $K$ previous tasks and $D^T$ for the current task. 
Each task $D^i = \{(x, y)\}$ 
$(i = 1, ...,K)$ includes a set of observations. Note that,
$y$ is an array, including multiple objectives for configuration $x$.
For multi-objective problems with $p$ objectives, we propose to
transfer the knowledge about $p$ objectives individually. Thus, the
transfer learning of multiple objectives is turned into $p$ single-objective
transfer learning processes. For each dimension of the
objectives, we take the following transfer-learning technique:

1) We first train a surrogate model $M^i$ on $D^i$ for the $i$-th prior task
and $M^T$ on $D^T$; 

2) Based on $M^{1:K}$ and $M^T$, we then build a transfer learning surrogate by combining all base surrogates:
$M^{TL} = agg(\{M^1, ...,M^K,M^T \};w)$;

3) The surrogate $M^{TL}$ is used to guide the configuration search,
instead of the original $M^T$. 

Concretely, we use gPoE to combine the multiple base surrogates (agg), 
and the parameters $w$ are calculated based on the ranking of configurations, 
which reflects the similarity between the source tasks and the target task.


## Performance Comparison
We compare OpenBox with a competitive transfer learning baseline Vizier and a non-transfer baseline SMAC3. 
The average performance rank (the lower, the better) of each algorithm is shown in the following figure. 
For experimental setups, dataset information and more experimental results, 
please refer to our [published article](https://dl.acm.org/doi/abs/10.1145/3447548.3467061).


<img src="../../imgs/tl_lightgbm_75_rank_result.svg" width="70%" class="align-center">
