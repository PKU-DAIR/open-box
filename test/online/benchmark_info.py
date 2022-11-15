"""
This test script is used to get some useful information about a benchmark function.
Including: max/min values, lipschitz constant, a plot of the function.
It's useful when testing SafeOpt.
"""

# License: MIT
import os

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS  # export NUMEXPR_NUM_THREADS=1

import sys

sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Configuration

from openbox.benchmark.objective_functions.synthetic import Rosenbrock, SafetyConstrained

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range


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


def nd_linspace(start, end, shape):
    def it(s, e, m):
        if len(m) == 1:
            for i in np.linspace(s[0], e[0], m[0]):
                yield [i]
        else:
            for i in np.linspace(s[0], e[0], m[0]):
                for j in it(s[1:], e[1:], m[1:]):
                    yield [i] + j

    return np.array(list(it(start, end, shape)))


FUNCTION = SafetyConstrained(Rosenbrock(dim=2), h=10)
space = FUNCTION.config_space

dim = len(space.sample_configuration().keys())

SAMPLE = (50, 50) if dim == 2 else (2500,)

SAMPLES = 2500

if __name__ == "__main__":

    Xi = nd_linspace((0,) * dim, (1,) * dim, SAMPLE)
    X = [np.array([v for k, v in Configuration(space, vector=np.array(x)).items()]) for x in Xi]
    y = [FUNCTION(x, convert=False)['objs'][0] for x in X]

    lipschitz = 0

    print("T")

    for i in trange(SAMPLES):
        for j in range(i + 1, SAMPLES):
            lipschitz = max(lipschitz, np.abs((y[j] - y[i]) / np.linalg.norm(Xi[j] - Xi[i])))

    print("Lipschitz on normalized space is:", lipschitz)
    print("Function max value is:", np.max(y))
    print("Function min value is:", np.min(y))

    name = FUNCTION.__class__.__name__

    if dim == 1:
        plt.plot(X, y)

        plt.title(name)

        plt.savefig(f"tmp/{name}.jpg")
