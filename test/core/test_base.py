import pytest
import numpy as np
from openbox.core.base import build_acq_func, build_surrogate
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction

from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
from openbox.surrogate.lightgbm import LightGBM
from openbox.surrogate.skrf import RandomForestSurrogate
from openbox.surrogate.base.base_gp import BaseGP
from openbox.surrogate.tlbo.mfgpe import MFGPE
from openbox.surrogate.tlbo.rgpe import RGPE
from openbox.surrogate.tlbo.stacking_gpr import SGPR
from openbox.surrogate.tlbo.topo_variant3 import TOPO_V3


def test_build_acq_func(configspace_tiny, surrogate_model_abs, surrogate_model_gp):
    config_space = configspace_tiny
    model_abs = surrogate_model_abs
    model_gp = surrogate_model_gp

    for acq_func in ['ei', 'eips', 'logei', 'pi', 'lcb', 'lpei', 'mcei', 'parego']:
        result = build_acq_func(acq_func, model_abs)
        assert isinstance(result, AbstractAcquisitionFunction)

    for acq_func in ['mcparego']:
        result = build_acq_func(acq_func, [model_abs, model_abs])
        assert isinstance(result, AbstractAcquisitionFunction)

    for acq_func in ['ehvi', 'mcehvi']:
        result = build_acq_func(acq_func, [model_abs, model_abs], ref_point=[0.5, 0.5])
        assert isinstance(result, AbstractAcquisitionFunction)

    for acq_func in ['mcparegoc']:
        result = build_acq_func(acq_func, [model_abs, model_abs], [model_gp, model_gp])
        assert isinstance(result, AbstractAcquisitionFunction)

    for acq_func in ['ehvic', 'mcehvic']:
        result = build_acq_func(acq_func, [model_abs, model_abs], [model_gp, model_gp], ref_point=[0.5, 0.5])
        assert isinstance(result, AbstractAcquisitionFunction)

    for acq_func in ['mesmo', 'usemo']:
        result = build_acq_func(acq_func, [model_abs, model_abs], config_space=config_space)
        assert isinstance(result, AbstractAcquisitionFunction)

    for acq_func in ['mesmoc', 'mesmoc2']:
        result = build_acq_func(acq_func, [model_abs, model_abs], [model_gp, model_gp], config_space=config_space)
        assert isinstance(result, AbstractAcquisitionFunction)

    for acq_func in ['eic', 'mceic']:
        result = build_acq_func(acq_func, model_abs, [model_gp, model_gp])
        assert isinstance(result, AbstractAcquisitionFunction)


def test_build_surrogate(configspace_tiny, transfer_learning_history_single):
    config_space = configspace_tiny
    rng = np.random.RandomState(0)
    surrogate = build_surrogate('prf', config_space, rng)
    assert isinstance(surrogate, (RandomForestWithInstances, skRandomForestWithInstances))

    rng = np.random.RandomState(0)
    surrogate = build_surrogate('sk_prf', config_space, rng)
    assert isinstance(surrogate, skRandomForestWithInstances)

    surrogate = build_surrogate('lightgbm', config_space, rng)
    assert isinstance(surrogate, LightGBM)

    surrogate = build_surrogate('random_forest', config_space, rng)
    assert isinstance(surrogate, RandomForestSurrogate)

    surrogate = build_surrogate('gp', config_space, rng)
    assert isinstance(surrogate, BaseGP)

    transfer_learning_history = transfer_learning_history_single
    surrogate = build_surrogate('mfgpe', config_space, rng, transfer_learning_history)
    assert isinstance(surrogate, MFGPE)

    surrogate = build_surrogate('tlbo_rgpe_gp', config_space, rng, transfer_learning_history)
    assert isinstance(surrogate, RGPE)

    surrogate = build_surrogate('tlbo_sgpr_gp', config_space, rng, transfer_learning_history)
    assert isinstance(surrogate, SGPR)

    surrogate = build_surrogate('tlbo_topov3_gp', config_space, rng, transfer_learning_history)
    assert isinstance(surrogate, TOPO_V3)

    with pytest.raises(ValueError, match='.* for tlbo surrogate!'):
        build_surrogate('tlbo_invalid', config_space, rng, transfer_learning_history)

    with pytest.raises(ValueError, match='.* for surrogate!'):
        build_surrogate('invalid', config_space, rng)
