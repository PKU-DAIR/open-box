import math
import random
from typing import Callable, List, Union, Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from openbox.core.base import build_surrogate
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.history import Observation, History
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg


def nd_range(*args):
    """
    There should be some system function that have implemented this. However, I didn't find it.

    Example:
    list(nd_range(2,3)) -> [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    """
    size = args[0] if isinstance(args[0], tuple) else args
    if len(size) == 1:
        for i in range(size[0]):
            yield (i,)
    else:
        for i in range(size[0]):
            for j in nd_range(size[1:]):
                yield (i,) + j


class DefaultBeta:
    """
    The class to evaluate beta with given turn number t.
    The value of beta is used for predictive interval [mean - beta ** (1/2) * std, mean + beta ** (1/2) * std].

    b is a bound for RKHS norm of objective function f.
    sz is the number of sampled points.
    delta is the allowed failure probability.
    c is the constant where gamma = c * sz, as it's said that gamma has sublinear dependence of sz for our GP kernels.
    """

    def __init__(self, b: float, sz: int, delta: float, c: float = 1.0):
        self.b = b
        self.sz = sz
        self.delta = delta
        self.c = c

    def __call__(self, t: float):
        gamma = self.c * self.sz
        return 2 * self.b + 300 * gamma * (math.log(t / self.delta) ** 3)


class SetManager:
    """
    Maintain a set of n-d linspaced points. Currently, only support 1-d and 2-d.
    Use boolean arrays to determine whether they're in some sets (s, g, m, vis)
    Also stores their GP prediction info (upper, lower)
    """

    def __init__(self, config_space: ConfigurationSpace, size: Tuple[int]):
        self.config_space = config_space
        self.dim = len(size)
        self.size = size

        self.s_set = np.zeros(size, dtype=np.bool)  # Safe set
        self.g_set = np.zeros(size, dtype=np.bool)  # Expander set
        self.m_set = np.zeros(size, dtype=np.bool)  # Minimizer set

        self.vis_set = np.zeros(size, dtype=np.bool)  # Set of evaluated points. Added this to avoid repeated configs.

        self.upper_conf = np.ones(size, dtype=np.float64) * 1e100
        self.lower_conf = np.ones(size, dtype=np.float64) * -1e100

        self.tmp = np.zeros(size, dtype=int)

    def _presum(self):
        """
        Replace self.tmp with the presum array for self.tmp
        This makes axis-aligned range query & modification on 1-d & 2-d arrays O(1).
        """
        if self.dim == 1:
            for i in range(1, self.size[0]):
                self.tmp[i] += self.tmp[i - 1]
        else:
            for i in range(1, self.size[0]):
                self.tmp[i, :] += self.tmp[i - 1, :]
            for i in range(1, self.size[1]):
                self.tmp[:, i] += self.tmp[:, i - 1]

    def _query_range(self, l: Tuple[int], r: Tuple[int]):
        """
        Given index ranges l,r (where l,r could be 1-d or 2-d indexes),
         calculate the sum of all elements in an array with index l <= i <= r (element-wise).
        Before calling this function, the array should be copied to self.tmp and self._presum() should be called.
        """
        if self.dim == 1:
            return self.tmp[r] - (0 if l[0] == 0 else self.tmp[l[0] - 1])
        else:
            x1, y1 = l
            x2, y2 = r
            ans = self.tmp[x2, y2]
            if x1 > 0:
                ans -= self.tmp[x1 - 1, y2]
                if y1 > 0:
                    ans += self.tmp[x1 - 1, y1 - 1]
            if y1 > 0:
                ans -= self.tmp[x2, y1 - 1]
            return ans

    def _add_range(self, l: Tuple[int], r: Tuple[int]):
        """
        Given index ranges l,r (where l,r could be 1-d or 2-d indexes),
         add 1 to all elements in an array with index l <= i <= r (element-wise).
        Before calling this function, self.tmp must be cleared.
        After a sequence of calling this function, self._presum() should be called
         and self.tmp could be copied back to the original array.
        """
        if self.dim == 1:
            self.tmp[l] += 1
            if r[0] < self.size[0] - 1:
                self.tmp[r[0] + 1] -= 1
        else:
            x1, y1 = l
            x2, y2 = r
            self.tmp[x1, y1] += 1
            if x2 < self.size[0] - 1:
                self.tmp[x2 + 1, y1] -= 1
                if y2 < self.size[1] - 1:
                    self.tmp[x2 + 1, y2 + 1] += 1
            if y2 < self.size[1] - 1:
                self.tmp[x1, y2 + 1] -= 1

    def nearest(self, x0: np.ndarray):
        """
        Return the index of the nearest point to x0 among the sampled points.
        x0: normalized configuration array.
        """
        return tuple(int(x0[i] * (self.size[i] - 1) + 0.5) for i in range(self.dim))

    def update_bounds(self, i: Tuple[int], m: float, v: float, b: float):
        """
        This is called when a sampled point at index i is predicted on GP,
        and upper/lower confidence should be updated.
        b is the value of beta^(1/2).
        """
        i = tuple(i)
        self.upper_conf[i] = min(self.upper_conf[i], m + v * b)
        self.lower_conf[i] = max(self.lower_conf[i], m - v * b)

    def update_s_set(self, h: float, l: float):
        """
        Update safe set according to the safeopt process.
        """
        self.tmp.fill(0)
        for i in nd_range(self.size):
            if self.s_set[i]:
                maxd = (h - self.upper_conf[i]) / l
                # print(maxd)
                if maxd > 0:
                    t = self.dim ** 0.5
                    mn = tuple(max(math.ceil(i[j] - maxd * (self.size[j] - 1) / t), 0) for j in range(self.dim))
                    mx = tuple(min(math.floor(i[j] + maxd * (self.size[j] - 1) / t), self.size[j] - 1) for j in
                               range(self.dim))
                    self._add_range(mn, mx)

        self._presum()
        self.s_set |= (self.tmp > 0)

    def update_g_set(self, h: float, l: float):
        """
        Update expander set according to the safeopt process.
        """
        self.g_set.fill(False)
        self.tmp.fill(1)
        self.tmp -= self.s_set
        self._presum()

        for i in nd_range(self.size):
            if self.s_set[i]:
                maxd = (h - self.lower_conf[i]) / l
                if maxd > 0:
                    t = self.dim ** 0.5
                    mn = tuple(max(math.ceil(i[j] - maxd * (self.size[j] - 1) / t), 0) for j in range(self.dim))
                    mx = tuple(min(math.floor(i[j] + maxd * (self.size[j] - 1) / t), self.size[j] - 1) for j in
                               range(self.dim))

                    if self._query_range(mn, mx) > 0:
                        self.g_set[i] = True

    def update_m_set(self, minu: float):
        """
        Update minimizer set according to the safeopt process.
        """
        self.m_set = self.s_set & (self.lower_conf <= minu)

    def get_array(self, coord: Tuple[int]):
        """
        Get the array (normalized configuration array) of a sampled points at some index.
        """
        if isinstance(coord, Configuration):
            coord = coord.get_array()

        return np.array(list(coord[i] / float(self.size[i] - 1) for i in range(self.dim)))

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the arrays of all config in s_set and their coordinates. For GP prediction of each turn.
        """
        arrays = []
        ids = []
        for i in nd_range(self.size):
            if self.s_set[i]:
                arrays.append(self.get_array(i))
                ids.append(i)

        return np.array(arrays), np.array(ids)

    def get_config(self, coord: Tuple[int]):
        return Configuration(self.config_space, vector=self.get_array(coord))


class SafeOptAdvisor:

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space: ConfigurationSpace,
                 num_objectives=1,
                 num_constraints=1,
                 task_id='OpenBox',
                 random_state=None,

                 surrogate: Union[str, AbstractModel] = 'gp',

                 sample_size: Union[int, Tuple] = 40000,
                 seed_set: Union[None, List[Configuration], np.ndarray] = None,

                 lipschitz: float = 20.0,  # The Lipschitz constant of the function.
                 threshold: float = 1.0,  # The h-value where the constraint is y - h.
                 beta: Union[float, Callable[[float], float]] = 2.0  # The beta used in original code.

                 ):
        self.num_objectives = num_objectives
        # May support multi-obj in the future.
        assert self.num_objectives == 1

        self.num_constraints = num_constraints
        # Let's assume that the only constraint is x - h.
        assert self.num_constraints == 1

        self.config_space = config_space
        self.dim = len(config_space.keys())
        self.rng = check_random_state(random_state)
        self.task_id = task_id

        if isinstance(surrogate, str):
            self.objective_surrogate: AbstractModel = build_surrogate(surrogate, config_space, self.rng or random, None)
        elif isinstance(surrogate, AbstractModel):
            self.objective_surrogate = surrogate

        if isinstance(sample_size, int):
            sample_size = (int(sample_size ** (1 / self.dim)),) * self.dim

        self.sets = SetManager(self.config_space, sample_size)

        if seed_set is None:
            raise ValueError("Seed set must not be None!")
        elif isinstance(seed_set, list):
            self.seed_set = seed_set
        else:
            self.seed_set = [Configuration(config_space, vector=seed_set[i]) for i in range(seed_set.shape[0])]

        for x in self.seed_set:
            self.sets.s_set[self.sets.nearest(x.get_array())] = True

        self.threshold = threshold
        self.lipschitz = lipschitz

        if callable(beta):
            self.beta = beta
        else:
            self.beta = lambda t: beta

        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=0,
            config_space=config_space, ref_point=None, meta_info=None,  # todo: add meta info
        )

        arrays = self.sets.get_arrays()[0]

        # This list stores the configs that the advisor expects to be evaluated.
        # Useful because we want to evaluate the seed set.
        # get_suggestion() fills this if it's empty.
        # update_observation() removes configs that is evaluated.
        self.to_eval = [Configuration(config_space, vector=arrays[i]) for i in
                                             range(arrays.shape[0])]

        self.current_turn = 0

    def debug(self, arr):
        if self.dim == 1:
            s = "".join("_" if not i else "|" for i in arr)
            print(s)
        else:
            print("-" * (self.sets.size[1] + 2))
            for i in range(self.sets.size[0]):
                print("|" + "".join("#" if j else " " for j in arr[i]) + "|")
            print("-" * (self.sets.size[1] + 2))

    def get_suggestion(self):
        if len(self.to_eval) == 0:  # If self.to_eval has some configs, it's the seed set. Evaluate them first.

            # Train GP model
            X = self.history.get_config_array(transform='scale')
            Y = self.history.get_objectives(transform='infeasible')
            self.objective_surrogate.train(X, Y[:, 0] if Y.ndim == 2 else Y)

            # Calculate beta^(-1/2)
            self.current_turn += 1
            beta_sqrt = self.beta(self.current_turn) ** 0.5

            # Re-predict all points on safe_set, and update their confidence bounds.
            arrays, ids = self.sets.get_arrays()
            mean, var = self.objective_surrogate.predict(arrays)
            for i in range(ids.shape[0]):
                self.sets.update_bounds(ids[i], mean[i].item(), var[i].item(), beta_sqrt)

            # According to the safeopt process, update the safe, expander, minimizer set.
            self.sets.update_s_set(self.threshold, self.lipschitz)
            self.sets.update_g_set(self.threshold, self.lipschitz)
            minu = np.min(self.sets.upper_conf[self.sets.s_set])
            self.sets.update_m_set(minu)

            # Find the point in the union of expander & minimizer set
            # with maximum uncertainty and have not been evaluated.
            retx = None
            maxv = -1e100

            for i in nd_range(self.sets.size):
                condition = (self.sets.g_set[i] or self.sets.m_set[i]) and not self.sets.vis_set[i]
                if condition:
                    w = self.sets.upper_conf[i] - self.sets.lower_conf[i]
                    if w > maxv:
                        maxv = w
                        retx = i

            if retx is not None:  # If such point is found, return that.
                self.to_eval = [self.sets.get_config(retx)]
            else:
                # Otherwise, select a random point by some heuristics.
                # This should be very rare. It's just to avoid error.

                # Select a random point in the safe set.
                possibles = self.sets.s_set & ~self.sets.vis_set

                # If doesn't exist, select a random point near the safe set.
                if not np.any(possibles):
                    temp = self.sets.vis_set
                    temp[1:] |= temp[:-1]
                    temp[:-1] |= temp[1:]

                    if self.dim == 2:
                        temp[:, 1:] |= temp[:, :-1]
                        temp[:, :-1] |= temp[:, 1:]

                    possibles = temp & ~self.sets.vis_set

                # If doesn't exist, just select a random point that have not been evaluated.
                if not np.any(possibles):
                    possibles = ~self.sets.vis_set

                coords = np.array(list(nd_range(self.sets.size)))[possibles.flatten()]

                self.to_eval = [self.sets.get_config(coords[self.rng.randint(0, coords.shape[0])])]

        return self.to_eval[0]

    def update_observation(self, observation: Observation):

        if observation.config in self.to_eval:
            self.to_eval.remove(observation.config)
            self.sets.vis_set[self.sets.nearest(observation.config.get_array())] = True

        observation.constraints = None

        return self.history.update_observation(observation)

    def get_history(self):
        return self.history
