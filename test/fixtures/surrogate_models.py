import pytest
import numpy as np
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.util_funcs import get_types
from openbox.surrogate.base.gp_base_prior import LognormalPrior
from openbox.surrogate.base.gp_kernels import ConstantKernel

@pytest.fixture
def surrogate_model_abs(configspace_tiny):
    config_space = configspace_tiny
    types, bounds = get_types(config_space)
    return AbstractModel(types=types, bounds=bounds)


@pytest.fixture
def surrogate_model_gp(configspace_tiny):
    config_space = configspace_tiny
    rng = np.random.RandomState(0)
    types, bounds = get_types(config_space)
    return create_gp_model(model_type='gp',
                           config_space=config_space,
                           types=types,
                           bounds=bounds,
                           rng=rng)


@pytest.fixture
def cons_kernel():
    rng = np.random.RandomState(0)

    return ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
    )
