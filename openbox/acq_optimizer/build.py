
def build_acq_optimizer(func_str='local_random', config_space=None, rng=None):
    assert config_space is not None
    func_str = func_str.lower()

    if func_str == 'local_random':
        from .basic_maximizer import InterleavedLocalAndRandomSearchMaximizer
        optimizer = InterleavedLocalAndRandomSearchMaximizer
    elif func_str == 'random_scipy':
        from .basic_maximizer import RandomScipyMaximizer
        optimizer = RandomScipyMaximizer
    elif func_str == 'scipy_global':
        from .basic_maximizer import ScipyGlobalMaximizer
        optimizer = ScipyGlobalMaximizer
    elif func_str == 'mesmo_optimizer':
        from .basic_maximizer import MESMO_Maximizer
        optimizer = MESMO_Maximizer
    elif func_str == 'usemo_optimizer':
        from .basic_maximizer import USeMO_Maximizer
        optimizer = USeMO_Maximizer
    elif func_str == 'cma_es':
        from .basic_maximizer import CMAESMaximizer
        optimizer = CMAESMaximizer
    elif func_str == 'batchmc':
        from .basic_maximizer import batchMCMaximizer
        optimizer = batchMCMaximizer
    elif func_str == 'staged_batch_scipy':
        from .basic_maximizer import StagedBatchScipyMaximizer
        optimizer = StagedBatchScipyMaximizer
    else:
        raise ValueError('Invalid string %s for acq_optimizer!' % func_str)

    return optimizer(config_space=config_space, rng=rng)
