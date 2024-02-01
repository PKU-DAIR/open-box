import pytest
from openbox.acq_optimizer import build_acq_optimizer
from openbox.acq_optimizer import InterleavedLocalAndRandomSearchMaximizer
from openbox.acq_optimizer import RandomScipyMaximizer
from openbox.acq_optimizer import ScipyGlobalMaximizer
from openbox.acq_optimizer import MESMO_Maximizer
from openbox.acq_optimizer import USeMO_Maximizer
from openbox.acq_optimizer import CMAESMaximizer
from openbox.acq_optimizer import batchMCMaximizer
from openbox.acq_optimizer import StagedBatchScipyMaximizer


def test_build_maximizer(configspace_tiny):
    config_space = configspace_tiny
    maximizer = build_acq_optimizer('local_random', config_space)
    assert isinstance(maximizer, InterleavedLocalAndRandomSearchMaximizer)

    maximizer = build_acq_optimizer('random_scipy', config_space)
    assert isinstance(maximizer, RandomScipyMaximizer)

    maximizer = build_acq_optimizer('scipy_global', config_space)
    assert isinstance(maximizer, ScipyGlobalMaximizer)

    maximizer = build_acq_optimizer('MESMO_Optimizer', config_space)
    assert isinstance(maximizer, MESMO_Maximizer)

    maximizer = build_acq_optimizer('USeMO_Optimizer', config_space)
    assert isinstance(maximizer, USeMO_Maximizer)

    maximizer = build_acq_optimizer('cma_es', config_space)
    assert isinstance(maximizer, CMAESMaximizer)

    maximizer = build_acq_optimizer('batchmc', config_space)
    assert isinstance(maximizer, batchMCMaximizer)

    maximizer = build_acq_optimizer('staged_batch_scipy', config_space)
    assert isinstance(maximizer, StagedBatchScipyMaximizer)

    with pytest.raises(ValueError):
        build_acq_optimizer('invalid', config_space)

