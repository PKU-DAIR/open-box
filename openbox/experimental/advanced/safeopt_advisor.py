import math
import random
from typing import Callable, List, Union, Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration

from openbox.core.base import build_surrogate
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.history import Observation, History
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg

from openbox.core.base_advisor import BaseAdvisor


def nd_range(shape, prefix=()):
    if len(shape) == 0:
        yield prefix
    else:
        for i in range(shape[0]):
            yield from nd_range(shape[1:], prefix + (i,))


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

        self.s_set = np.full(size, False)  # Safe set
        self.g_set = np.full(size, False)  # Expander set
        self.m_set = np.full(size, False)  # Minimizer set

        self.vis_set = np.full(size, False)  # Set of evaluated points. Added this to avoid repeated configs.

        self.upper_conf = np.full(size, 1e100)
        self.lower_conf = np.full(size, -1e100)


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
        for i in nd_range(self.size):
            if self.s_set[i]:
                maxd = (h - self.upper_conf[i]) / l
                if maxd > 0:
                    t = self.dim ** 0.5
                    new_s_ranges_slices = tuple(
                        slice(
                            max(math.ceil( i[j] - maxd * (self.size[j] - 1) / t), 0               ),
                            min(math.floor(i[j] + maxd * (self.size[j] - 1) / t), self.size[j] - 1) 
                        )
                        for j in range(self.dim)
                    )

                    self.s_set[new_s_ranges_slices] = True

    def update_g_set(self, h: float, l: float):
        """
        Update expander set according to the safeopt process.
        """
        self.g_set.fill(False)
        
        tmp = np.full(self.s_set.shape, True)
        tmp ^= self.s_set

        for i in nd_range(self.size):
            if self.s_set[i]:
                maxd = (h - self.lower_conf[i]) / l
                if maxd > 0:
                    t = self.dim ** 0.5
                    check_ranges_slices = tuple(
                        slice(
                            max(math.ceil( i[j] - maxd * (self.size[j] - 1) / t), 0               ),
                            min(math.floor(i[j] + maxd * (self.size[j] - 1) / t), self.size[j] - 1) 
                        )
                        for j in range(self.dim)
                    )

                    if np.any(tmp[check_ranges_slices]):
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


class SafeOptAdvisor(BaseAdvisor):

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
        self, 
        config_space: ConfigurationSpace,
        num_objectives=1,
        num_constraints=1,
        ref_point=None,
        output_dir='logs',
        
        task_id='OpenBox',
        random_state=None,

        logger_kwargs: dict = None,

        surrogate: Union[str, AbstractModel] = 'gp',

        sample_size: Union[int, Tuple] = 40000,
        seed_set: Union[None, List[Configuration], np.ndarray] = None,

        lipschitz: float = 20.0,  # The Lipschitz constant of the function.
        threshold: float = 1.0,  # The h-value where the constraint is y - h.
        beta: Union[float, Callable[[float], float]] = 2.0  # The beta used in original code.

    ):
        super().__init__(
            config_space=config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            ref_point=ref_point,
            output_dir=output_dir,
            task_id=task_id,
            random_state=random_state,
            logger_kwargs=logger_kwargs,
        )
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
            sample_size = np.array([int(sample_size ** (1 / self.dim) + 1e-5)] * self.dim)

        sets = SetManager(self.config_space, sample_size)

        if seed_set is None:
            raise ValueError("Seed set must not be None!")
        elif isinstance(seed_set, list):
            if all(isinstance(item, Configuration) for item in seed_set):
                self.seed_set = seed_set
            elif all(isinstance(item, dict) for item in seed_set):
                self.seed_set = [Configuration(config_space, item) for item in seed_set]
            elif all(isinstance(item, (list, np.ndarray)) for item in seed_set):
                self.seed_set = [Configuration(config_space, vector=item) for item in seed_set]
        elif isinstance(seed_set, np.ndarray):
            if seed_set.ndim == 2 and seed_set.shape[1] == self.dim:
                self.seed_set = [Configuration(config_space, vector=seed_set[i]) for i in range(seed_set.shape[0])]
            else:
                raise ValueError("Invalid shape of seed set!")
        else:
            raise ValueError("Invalid type of seed set!")

        for x in self.seed_set:
            sets.s_set[sets.nearest(x.get_array())] = True

        self.threshold = threshold
        self.lipschitz = lipschitz

        if callable(beta):
            self.beta = beta
        else:
            self.beta = lambda t: beta

        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=0,
            config_space=config_space, ref_point=None, meta_info={'set_manager': sets},  # todo: add meta info
        )

        arrays = sets.get_arrays()[0]

        # This list stores the configs that the advisor expects to be evaluated.
        # Useful because we want to evaluate the seed set.
        # get_suggestion() fills this if it's empty.
        # update_observation() removes configs that is evaluated.
        self.to_eval = [Configuration(config_space, vector=arrays[i])
                        for i in range(arrays.shape[0])]

        self.current_turn = 0

    def get_suggestion(self, history: History = None):
        if history is None:
            history = self.history

        sets = history.meta_info['set_manager']

        if len(self.to_eval) == 0:  # If self.to_eval has some configs, it's the seed set. Evaluate them first.

            # Train GP model
            X = history.get_config_array(transform='scale')
            Y = history.get_objectives(transform='infeasible')
            self.objective_surrogate.train(X, Y[:, 0] if Y.ndim == 2 else Y)

            # Calculate beta^(-1/2)
            self.current_turn += 1
            beta_sqrt = self.beta(self.current_turn) ** 0.5

            # Re-predict all points on safe_set, and update their confidence bounds.
            arrays, ids = sets.get_arrays()
            mean, var = self.objective_surrogate.predict(arrays)
            for i in range(ids.shape[0]):
                sets.update_bounds(ids[i], mean[i].item(), var[i].item(), beta_sqrt)

            # According to the safeopt process, update the safe, expander, minimizer set.
            sets.update_s_set(self.threshold, self.lipschitz)
            sets.update_g_set(self.threshold, self.lipschitz)
            minu = np.min(sets.upper_conf[sets.s_set])
            sets.update_m_set(minu)


            # Find the point in the union of expander & minimizer set
            # with maximum uncertainty and have not been evaluated.
            retx = None
            maxv = -1e100

            for i in nd_range(sets.size):
                condition = (sets.g_set[i] or sets.m_set[i]) and not sets.vis_set[i]
                if condition:
                    w = sets.upper_conf[i] - sets.lower_conf[i]
                    if w > maxv:
                        maxv = w
                        retx = i

            if retx is not None:  # If such point is found, return that.
                self.to_eval = [sets.get_config(retx)]
            else:
                # Otherwise, select a random point by some heuristics.
                # This should be very rare. It's just to avoid error.

                # Select a random point in the safe set.
                possibles = sets.s_set & ~sets.vis_set

                # If doesn't exist, select a random point near the safe set.
                if not np.any(possibles):
                    temp = sets.vis_set
                    temp[1:] |= temp[:-1]
                    temp[:-1] |= temp[1:]

                    if self.dim == 2:
                        temp[:, 1:] |= temp[:, :-1]
                        temp[:, :-1] |= temp[:, 1:]

                    possibles = temp & ~sets.vis_set

                # If doesn't exist, just select a random point that have not been evaluated.
                if not np.any(possibles):
                    possibles = ~sets.vis_set

                coords = np.array(list(nd_range(sets.size)))[possibles.flatten()]

                self.to_eval = [sets.get_config(coords[self.rng.randint(0, coords.shape[0])])]

        return self.to_eval[0]

    def update_observation(self, observation: Observation):
        sets = self.history.meta_info['set_manager']

        if observation.config in self.to_eval:
            self.to_eval.remove(observation.config)
            sets.vis_set[sets.nearest(observation.config.get_array())] = True

        observation.constraints = None

        return self.history.update_observation(observation)

