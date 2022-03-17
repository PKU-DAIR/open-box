# License: MIT

r"""
Helper utilities for constructing scalarizations adapted from botorch.

References

.. [Knowles2005]
    J. Knowles, "ParEGO: a hybrid algorithm with on-line landscape approximation
    for expensive multiobjective optimization problems," in IEEE Transactions
    on Evolutionary Computation, vol. 10, no. 1, pp. 50-66, Feb. 2006.
"""

import numpy as np


def get_chebyshev_scalarization(
    weights: np.ndarray, Y: np.ndarray, alpha: float = 0.05
):
    r"""Construct an augmented Chebyshev scalarization.

    Outcomes are first normalized to [0,1] and then an augmented
    Chebyshev scalarization is applied.

    Augmented Chebyshev scalarization:
        objective(y) = max(w * y) + alpha * sum(w * y)

    Note: this assumes minimization.

    See [Knowles2005]_ for details.

    This scalarization can be used with ExpectedImprovement to implement ParEGO
    as proposed in [Daulton2020qehvi]_.

    Args:
        weights: A `m`-dim array of weights.
        Y: A `n x m`-dim array of observed outcomes, which are used for
            scaling the outcomes to [0,1].
        alpha: Parameter governing the influence of the weighted sum term. The
            default value comes from [Knowles2005]_.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = np.array([0.75, 0.25])
        >>> transform = get_chebyshev_scalarization(weights, Y)
    """
    if weights.shape != Y.shape[-1:]:
        raise Exception(
            "weights must be an `m`-dim array where Y is `... x m`."
            "Got shapes %s and %s." % (str(weights.shape), str(Y.shape))
        )
    Y_bounds = np.stack([Y.min(axis=-2), Y.max(axis=-2)])

    def obj(Y: np.array) -> np.array:
        # scale to [0,1]
        Y_normalized = (Y - Y_bounds[0]) / (Y_bounds[1] - Y_bounds[0])
        product = weights * Y_normalized
        return product.max(axis=-1) + alpha * product.sum(axis=-1)

    return obj
