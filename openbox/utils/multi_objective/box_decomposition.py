# License: MIT

r"""Algorithms for partitioning the non-dominated space into rectangles.

Code is adapted from botorch.

References

.. [Couckuyt2012]
    I. Couckuyt, D. Deschrijver and T. Dhaene, "Towards Efficient
    Multiobjective Optimization: Multiobjective statistical criterions,"
    2012 IEEE Congress on Evolutionary Computation, Brisbane, QLD, 2012,
    pp. 1-8.

"""

from typing import Optional
import numpy as np

from openbox.utils.multi_objective.pareto import is_non_dominated


class NondominatedPartitioning(object):
    r"""A class for partitioning the non-dominated space into hyper-cells.

    Note: this assumes minimization.

    Note: it is only feasible to use this algorithm to compute an exact
    decomposition of the non-dominated space for `m<5` objectives (alpha=0.0).

    The alpha parameter can be increased to obtain an approximate partitioning
    faster. The `alpha` is a fraction of the total hypervolume encapsuling the
    entire pareto set. When a hypercell's volume divided by the total hypervolume
    is less than `alpha`, we discard the hypercell. See Figure 2 in
    [Couckuyt2012]_ for a visual representation.

    This numpy implementation is adapted from botorch, which is adapted from
    https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/pareto.py.
    """

    def __init__(
        self,
        num_objs: int,
        Y: Optional[np.ndarray] = None,
        alpha: float = 0.0,
        eps: Optional[float] = None,
    ) -> None:
        """Initialize NondominatedPartitioning.

        Args:
            num_objs: The number of objective functions
            Y: A `n x m`-dim array
            alpha: a thresold fraction of total volume used in an approximate
                decomposition.
            eps: a small value for numerical stability
        """
        self.alpha = alpha
        self.num_objs = num_objs
        self._eps = eps
        if Y is not None:
            self.update(Y=Y)

    @property
    def eps(self) -> float:
        if self._eps is not None:
            return self._eps
        try:
            return 1e-6 if self._pareto_Y.dtype == np.float32 else 1e-8
        except AttributeError:
            return 1e-6

    @property
    def pareto_Y(self) -> np.ndarray:
        r"""This returns the non-dominated set assuming minimization.

        Returns:
            A `n_pareto x m`-dim array of outcomes.
        """
        if not hasattr(self, "_pareto_Y"):
            raise Exception("pareto_Y has not been initialized")
        return self._pareto_Y

    def _update_pareto_Y(self) -> bool:
        r"""Update the non-dominated front."""
        non_dominated_mask = is_non_dominated(self.Y)
        pf = self.Y[non_dominated_mask]
        # sort by first objective
        new_pareto_Y = pf[np.argsort(pf[:, 0])]
        if not hasattr(self, "_pareto_Y") or not np.equal(
            new_pareto_Y, self._pareto_Y
        ):
            self._pareto_Y = new_pareto_Y
            return True
        return False

    def update(self, Y: np.ndarray) -> None:
        r"""Update non-dominated front and decomposition.

        Args:
            Y: A `n x m`-dim array of outcomes.
        """
        self.Y = Y
        is_new_pareto = self._update_pareto_Y()
        # Update decomposition if the pareto front changed
        if is_new_pareto:
            if self.num_objs > 2:
                self.binary_partition_non_dominated_space()
            else:
                self.partition_non_dominated_space_2d()

    def binary_partition_non_dominated_space(self):
        r"""Partition the non-dominated space into disjoint hypercells.

        This method works for an arbitrary number of outcomes, but is
        less efficient than `partition_non_dominated_space_2d` for the
        2-outcome case.
        """
        # Extend pareto front with the ideal and anti-ideal point
        ideal_point = self._pareto_Y.min(axis=0, keepdims=True) - 1
        anti_ideal_point = self._pareto_Y.max(axis=0, keepdims=True) + 1

        aug_pareto_Y = np.concatenate([ideal_point, self._pareto_Y, anti_ideal_point], axis=0)
        # The binary parititoning algorithm uses indices the augmented pareto front.
        aug_pareto_Y_idcs = self._get_augmented_pareto_front_indices()

        # Initialize one cell over entire pareto front
        cell = np.zeros((2, self.num_objs), dtype=int)
        cell[1] = aug_pareto_Y_idcs.shape[0] - 1
        stack = [cell]
        total_volume = (anti_ideal_point - ideal_point).prod()

        # hypercells contains the indices of the (augmented) pareto front
        # that specify that bounds of the each hypercell.
        # It is a `2 x num_cells x num_objs`-dim array
        self.hypercells = np.empty((2, 0, self.num_objs))
        outcome_idxr = np.arange(self.num_objs)

        # Use binary partitioning
        while len(stack) > 0:
            cell = stack.pop()
            cell_bounds_pareto_idcs = aug_pareto_Y_idcs[cell, outcome_idxr]
            cell_bounds_pareto_values = aug_pareto_Y[
                cell_bounds_pareto_idcs, outcome_idxr
            ]
            # Check cell bounds
            # - if cell upper bound is better than pareto front on all outcomes:
            #   - accept the cell
            # - elif cell lower bound is better than pareto front on all outcomes:
            #   - this means the cell overlaps the pareto front. Divide the cell along
            #   - its longest edge.
            if (
                ((cell_bounds_pareto_values[1] - self.eps) < self._pareto_Y)
                .any(axis=1)
                .all()
            ):
                # Cell is entirely non-dominated
                self.hypercells = np.concatenate(
                    [self.hypercells, np.expand_dims(cell_bounds_pareto_idcs, 1)], axis=1
                )
            elif (
                ((cell_bounds_pareto_values[0] + self.eps) < self._pareto_Y)
                .any(axis=1)
                .all()
            ):
                # The cell overlaps the pareto front
                # compute the distance (in integer indices)
                idx_dist = cell[1] - cell[0]
                any_not_adjacent = (idx_dist > 1).any()
                cell_volume = (
                    (cell_bounds_pareto_values[1] - cell_bounds_pareto_values[0])
                    .prod(axis=-1)
                )

                # Only divide a cell when it is not composed of adjacent indices
                # and the fraction of total volume is above the approximation
                # threshold fraction
                if (
                    any_not_adjacent
                    and ((cell_volume / total_volume) > self.alpha).all()
                ):
                    # Divide the test cell over its largest dimension
                    # largest (by index length)
                    longest_dim = np.argmax(idx_dist)
                    length = idx_dist[longest_dim]

                    new_length1 = int(round(length / 2.0))
                    new_length2 = length - new_length1

                    # Store divided cells
                    # cell 1: subtract new_length1 from the upper bound of the cell
                    # cell 2: add new_length2 to the lower bound of the cell
                    for bound_idx, length_delta in (
                        (1, -new_length1),
                        (0, new_length2),
                    ):
                        new_cell = cell.copy()
                        new_cell[bound_idx, longest_dim] += length_delta
                        stack.append(new_cell)

    def partition_non_dominated_space_2d(self) -> None:
        r"""Partition the non-dominated space into disjoint hypercells.

        This direct method works for `m=2` outcomes.
        """
        if self.num_objs != 2:
            raise Exception(
                "partition_non_dominated_space_2d requires 2 outputs, "
                "but num_objs=%d" % self.num_objs
            )
        pf_ext_idx = self._get_augmented_pareto_front_indices()
        range_pf_plus1 = np.arange(
            self._pareto_Y.shape[0] + 1
        )
        lower = np.stack([range_pf_plus1, np.zeros_like(range_pf_plus1)], axis=-1)
        upper = np.stack(
            [range_pf_plus1 + 1, pf_ext_idx[-range_pf_plus1 - 1, -1]], axis=-1
        )
        self.hypercells = np.stack([lower, upper], axis=0)

    def _get_augmented_pareto_front_indices(self) -> np.ndarray:
        r"""Get indices of augmented pareto front."""
        pf_idx = np.argsort(self._pareto_Y, axis=0)
        return np.concatenate(
            [
                np.zeros(
                    (1, self.num_objs), dtype=int
                ),
                # Add 1 because index zero is used for the ideal point
                pf_idx + 1,
                np.full(
                    (1, self.num_objs),
                    self._pareto_Y.shape[0] + 1,
                    dtype=int
                ),
            ],
            axis=0,
        )

    def get_hypercell_bounds(self, ref_point: np.ndarray) -> np.ndarray:
        r"""Get the bounds of each hypercell in the decomposition.

        Args:
            ref_point: A `m`-dim array containing the reference point.

        Returns:
            A `2 x num_cells x num_objs`-dim array containing the
                lower and upper vertices bounding each hypercell.
        """
        aug_pareto_Y = np.concatenate(
            [
                # -inf is the lower bound of the non-dominated space
                np.full(
                    (1, self.num_objs),
                    float("-inf")
                ),
                self._pareto_Y,
                np.expand_dims(ref_point, 0),
            ],
            axis=0,
        )
        return self._get_hypercell_bounds(aug_pareto_Y=aug_pareto_Y)

    def _get_hypercell_bounds(self, aug_pareto_Y: np.ndarray) -> np.ndarray:
        r"""Get the bounds of each hypercell in the decomposition.

        Args:
            aug_pareto_Y: A `n_pareto + 2 x m`-dim array containing
            the augmented pareto front.

        Returns:
            A `2 x num_cells x num_objs`-dim array containing the
                lower and upper vertices bounding each hypercell.
        """
        num_cells = self.hypercells.shape[1]
        outcome_idxr = np.tile(np.arange(self.num_objs), num_cells)

        # this array is 2 x (num_cells *num_objs) x 2
        # the batch dim corresponds to lower/upper bound
        cell_bounds_idxr = np.stack(
            [self.hypercells.reshape(2, -1),
             np.tile(outcome_idxr[None, :], (2, 1))],
            axis=-1,
        ).astype(int)

        def chunk(arr, chunks, axis):
            num_to_chunk = arr.shape[axis]
            chunk_size = np.ceil(num_to_chunk / chunks).astype(int)
            real_chunks = np.ceil(num_to_chunk / chunk_size).astype(int)
            return tuple(np.take(arr, indices=range(i*chunk_size, (i+1)*chunk_size), axis=axis) for i in range(real_chunks))

        indexers = chunk(cell_bounds_idxr, self.num_objs, axis=-1)
        cell_bounds_values = aug_pareto_Y[indexers].reshape(2, -1, self.num_objs)
        return cell_bounds_values

    def compute_hypervolume(self, ref_point: np.ndarray) -> float:
        r"""Compute the hypervolume for the given reference point.

        Note: This assumes minimization.

        This method computes the hypervolume of the non-dominated space
        and computes the difference between the hypervolume between the
        ideal point and hypervolume of the non-dominated space.

        Note there are much more efficient alternatives for computing
        hypervolume when m > 2 (which do not require partitioning the
        non-dominated space). Given such a partitioning, this method
        is quite fast.

        Args:
            ref_point: A `m`-dim array containing the reference point.

        Returns:
            The dominated hypervolume.
        """
        if (self._pareto_Y <= ref_point).any():
            raise ValueError(
                "The reference point must be smaller than all pareto_Y values."
            )
        ideal_point = self._pareto_Y.min(axis=0, keepdims=True).values
        ref_point = np.expand_dims(ref_point, 0)
        aug_pareto_Y = np.concatenate([ideal_point, self._pareto_Y, ref_point], axis=0)
        cell_bounds_values = self._get_hypercell_bounds(aug_pareto_Y=aug_pareto_Y)
        total_volume = (ref_point - ideal_point).prod()
        non_dom_volume = (
            (cell_bounds_values[1] - cell_bounds_values[0]).prod(axis=-1).sum()
        )
        return total_volume - non_dom_volume
