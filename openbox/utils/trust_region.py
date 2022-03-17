# License: MIT

import numpy as np


class TurboState(object):
    """
    Trust region Bayesian optimization state.

    Only supports single objective.
    """

    dim: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __init__(self, dim):
        self.dim = dim
        self.failure_tolerance = np.ceil(max([4.0, float(self.dim)]))

    def update(self, y_next):
        if y_next < self.best_value - 1e-3 * np.abs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = np.min([2.0 * self.length, self.length_max])
            self.success_counter = 0
            print('-'*30)
            print('Expand!')
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.success_counter = 0
            print('-'*30)
            print('Shrink!')

        self.best_value = np.min([self.best_value, y_next])
        if self.length < self.length_min:
            self.restart_triggered = True
