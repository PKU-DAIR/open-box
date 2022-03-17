# License: MIT

import sys
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.acquisition_function import *
from openbox.utils.util_funcs import get_types


class Observation(object):
    def __init__(self, config, objs, constraints=None, trial_state=SUCCESS, elapsed_time=None):
        self.config = config
        self.objs = objs
        self.constraints = constraints
        self.trial_state = trial_state
        self.elapsed_time = elapsed_time

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Observation(" \
               + "config=%s" % self.config \
               + ", objs=%s" % self.objs \
               + (", constraints=%s" % self.constraints if self.constraints is not None else "") \
               + ", trial_state=%s" % self.trial_state \
               + (", elapsed_time=%s" % self.elapsed_time if self.elapsed_time is not None else "") \
               + ")"


acq_dict = {
    'ei': EI,
    'eips': EIPS,
    'logei': LogEI,
    'pi': PI,
    'lcb': LCB,
    'lpei': LPEI,
    'ehvi': EHVI,
    'ehvic': EHVIC,
    'mesmo': MESMO,
    'usemo': USeMO,     # todo single acq type
    'mcei': MCEI,
    'parego': EI,
    'mcparego': MCParEGO,
    'mcparegoc': MCParEGOC,
    'mcehvi': MCEHVI,
    'mcehvic': MCEHVIC,
    'eic': EIC,
    'mesmoc': MESMOC,
    'mesmoc2': MESMOC2,
    'mceic': MCEIC,
}


def build_acq_func(func_str='ei', model=None, constraint_models=None, **kwargs):
    func_str = func_str.lower()
    acq_func = acq_dict.get(func_str)
    if acq_func is None:
        raise ValueError('Invalid string %s for acquisition function!' % func_str)
    if constraint_models is None:
        return acq_func(model=model, **kwargs)
    else:
        return acq_func(model=model, constraint_models=constraint_models, **kwargs)


def build_optimizer(func_str='local_random', acq_func=None, config_space=None, rng=None):
    assert config_space is not None
    func_str = func_str.lower()

    if func_str == 'local_random':
        from openbox.acq_maximizer.ei_optimization import InterleavedLocalAndRandomSearch
        optimizer = InterleavedLocalAndRandomSearch
    elif func_str == 'random_scipy':
        from openbox.acq_maximizer.ei_optimization import RandomScipyOptimizer
        optimizer = RandomScipyOptimizer
    elif func_str == 'scipy_global':
        from openbox.acq_maximizer.ei_optimization import ScipyGlobalOptimizer
        optimizer = ScipyGlobalOptimizer
    elif func_str == 'mesmo_optimizer':
        from openbox.acq_maximizer.ei_optimization import MESMO_Optimizer
        optimizer = MESMO_Optimizer
    elif func_str == 'usemo_optimizer':
        from openbox.acq_maximizer.ei_optimization import USeMO_Optimizer
        optimizer = USeMO_Optimizer
    elif func_str == 'cma_es':
        from openbox.acq_maximizer.ei_optimization import CMAESOptimizer
        optimizer = CMAESOptimizer
    elif func_str == 'batchmc':
        from openbox.acq_maximizer.ei_optimization import batchMCOptimizer
        optimizer = batchMCOptimizer
    elif func_str == 'staged_batch_scipy':
        from openbox.acq_maximizer.ei_optimization import StagedBatchScipyOptimizer
        optimizer = StagedBatchScipyOptimizer
    else:
        raise ValueError('Invalid string %s for acq_maximizer!' % func_str)

    return optimizer(acquisition_function=acq_func,
                     config_space=config_space,
                     rng=rng)


def build_surrogate(func_str='gp', config_space=None, rng=None, history_hpo_data=None):
    assert config_space is not None
    func_str = func_str.lower()
    types, bounds = get_types(config_space)
    seed = rng.randint(MAXINT)
    if func_str == 'prf':
        try:
            from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances
            return RandomForestWithInstances(types=types, bounds=bounds, seed=seed)
        except ModuleNotFoundError:
            from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
            print('[Build Surrogate] Use probabilistic random forest based on scikit-learn. For better performance, '
                  'please install pyrfr: '
                  'https://open-box.readthedocs.io/en/latest/installation/install_pyrfr.html')
            return skRandomForestWithInstances(types=types, bounds=bounds, seed=seed)

    elif func_str == 'sk_prf':
        from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
        return skRandomForestWithInstances(types=types, bounds=bounds, seed=seed)

    elif func_str == 'lightgbm':
        from openbox.surrogate.lightgbm import LightGBM
        return LightGBM(config_space, types=types, bounds=bounds, seed=seed)

    if func_str == 'random_forest':
        from openbox.surrogate.skrf import RandomForestSurrogate
        return RandomForestSurrogate(config_space, types=types, bounds=bounds, seed=seed)

    elif func_str.startswith('gp'):
        from openbox.surrogate.base.build_gp import create_gp_model
        return create_gp_model(model_type=func_str,
                               config_space=config_space,
                               types=types,
                               bounds=bounds,
                               rng=rng)
    elif func_str.startswith('mfgpe'):
        from openbox.surrogate.tlbo.mfgpe import MFGPE
        inner_surrogate_type = 'prf'
        return MFGPE(config_space, history_hpo_data, seed,
                     surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
    elif func_str.startswith('tlbo'):
        print('the current surrogate is', func_str)
        if 'rgpe' in func_str:
            from openbox.surrogate.tlbo.rgpe import RGPE
            inner_surrogate_type = func_str.split('_')[-1]
            return RGPE(config_space, history_hpo_data, seed,
                        surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
        elif 'sgpr' in func_str:
            from openbox.surrogate.tlbo.stacking_gpr import SGPR
            inner_surrogate_type = func_str.split('_')[-1]
            return SGPR(config_space, history_hpo_data, seed,
                        surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
        elif 'topov3' in func_str:
            from openbox.surrogate.tlbo.topo_variant3 import TOPO_V3
            inner_surrogate_type = func_str.split('_')[-1]
            return TOPO_V3(config_space, history_hpo_data, seed,
                           surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1)
        else:
            raise ValueError('Invalid string %s for tlbo surrogate!' % func_str)
    else:
        raise ValueError('Invalid string %s for surrogate!' % func_str)
