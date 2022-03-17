# License: MIT

import numpy as np


def is_non_dominated(Y: np.ndarray) -> np.ndarray:
    r"""Computes the non-dominated front.

    Note: this assumes minimization.

    Args:
        Y: a `(batch_shape) x n x m`-dim array of outcomes.

    Returns:
        A `(batch_shape) x n`-dim boolean array indicating whether
        each point is non-dominated.
    """
    expanded_shape = Y.shape[:-2] + Y.shape[-2:-1] + Y.shape[-2:]
    Y1 = np.broadcast_to(np.expand_dims(Y, -3), expanded_shape)
    Y2 = np.broadcast_to(np.expand_dims(Y, -2), expanded_shape)
    dominates = (Y1 <= Y2).all(axis=-1) & (Y1 < Y2).any(axis=-1)
    return ~(dominates.any(axis=-1))


def get_pareto_front(Y: np.ndarray):
    r"""
    Compute the pareto front.

    Note: this assumes minimization.

    Args:
        Y: a `N x m`-dim array of outcomes.

    Returns:
        pareto_front: a `k x m`-dim array.
    """
    pareto_front = np.arange(Y.shape[0])
    next_point_index = 0
    while next_point_index < len(Y):
        non_dominated_point_mask = np.any(Y < Y[next_point_index], axis=1)
        non_dominated_point_mask[next_point_index] = True
        pareto_front = pareto_front[non_dominated_point_mask]
        Y = Y[non_dominated_point_mask]
        next_point_index = np.sum(non_dominated_point_mask[:next_point_index]) + 1
    return pareto_front
