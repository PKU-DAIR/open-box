# Transfer Learning

When performing BBO, users often run tasks that are similar to
previous ones. This observation can be used to speed up the current task.
Compared with Vizier, which only provides limited transfer learning
functionality for single-objective BBO problems, OpenBox employs
a general transfer learning framework with the following
advantages:

1) Support for generalized black-box optimization problems;

2) Compatibility with most Bayesian optimization methods.

OpenBox takes as input observations from $𝐾 + 1$ tasks: $D^1$, ...,
$D^𝐾$ for $𝐾$ previous tasks and $D^𝑇$ for the current task. 
Each task $D^𝑖 = \{(𝒙, 𝒚)\}$ 
$(𝑖 = 1, ...,𝐾)$ includes a set of observations. Note that,
$𝒚$ is an array, including multiple objectives for configuration $𝒙$.
For multi-objective problems with $𝑝$ objectives, we propose to
transfer the knowledge about $𝑝$ objectives individually. Thus, the
transfer learning of multiple objectives is turned into $𝑝$ single-objective
transfer learning processes. For each dimension of the
objectives, we take the following transfer-learning technique:

1) We first train a surrogate model $𝑀^𝑖$ on $𝐷^𝑖$ for the $𝑖$-th prior task
and $𝑀^𝑇$ on $𝐷^𝑇$; 

2) Based on $𝑀^{1:𝐾}$ and $𝑀^𝑇$, we then build a transfer learning surrogate by combining all base surrogates:
$𝑀^{TL} = agg(\{𝑀^1, ...,𝑀^𝐾,𝑀^𝑇 \};w)$;

3) The surrogate $𝑀^{TL}$ is used to guide the configuration search,
instead of the original $𝑀^𝑇$. 

Concretely, we use gPoE to combine the multiple base surrogates (agg), 
and the parameters $w$ are calculated based on the ranking of configurations, 
which reflects the similarity between the source tasks and the target task.


## Performance Comparison
We compare OpenBox with a competitive transfer learning baseline Vizier and a non-transfer baseline SMAC3. 
The average performance rank (the lower, the better) of each algorithm is shown in the following figure. 
For experimental setups, dataset information and more experimental results, please refer to our [published article]().


<img src="../../imgs/tl_lightgbm_75_rank_result.svg" width="70%" class="align-center">
