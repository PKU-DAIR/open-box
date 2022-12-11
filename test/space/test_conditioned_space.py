import sys
sys.path.insert(0, '.')

import traceback
from openbox import space as sp
from openbox.utils.config_space import get_one_exchange_neighbourhood
from ConfigSpace import Configuration
import numpy as np
import pickle as pkl


verbose = True


def sample_condition(config):
    if verbose:
        print('- sample_condition called')
    # We want x1 >= x2 >= x3
    if config['x1'] < config['x2']:
        return False  # violate condition
    if config['x2'] < config['x3']:
        return False  # violate condition
    return True  # valid


if __name__ == '__main__':
    cs = sp.ConditionedSpace()
    for i in range(1, 4):
        cs.add_hyperparameter(sp.Int('x%d' % i, 0, 100, default_value=0))
    print(cs)
    cs.set_sample_condition(sample_condition)
    cs.seed(1234)
    print('space initialized.')

    print('\nsample 10 configs:')
    configs = cs.sample_configuration(10)
    print('\nvalidate configs:')
    for config in configs:
        config.is_valid_configuration()
        is_valid = sample_condition(config)
        print(config, is_valid)
        if not is_valid:
            raise ValueError('sampled invalid config')

    print('\ntest invalid values (dict)')
    try:
        config = Configuration(cs, values=dict(x1=1, x2=2, x3=3))
    except ValueError as e:
        traceback.print_exc()
        print('invalid values caught')
    print('\ntest invalid vector (np.array)')
    config = Configuration(cs, vector=np.array([0.1, 0.2, 0.3]))  # will not check in initialization with vector
    try:
        config.is_valid_configuration()
    except ValueError as e:
        traceback.print_exc()
        print('invalid vector caught')
    print('\ntest invalid neighbors')
    config = Configuration(cs, values=dict(x1=100, x2=50, x3=10))
    neighbors = list(get_one_exchange_neighbourhood(config, seed=2))
    print(len(neighbors), 'neighbors')
    assert all([sample_condition(config) for config in neighbors])

    verbose = False
    print('\nsample 10000 configs')
    configs = cs.sample_configuration(10000)
    assert all([sample_condition(config) for config in configs])

    print('\ntest pickle')
    bcs = pkl.dumps(cs)
    cs = pkl.loads(bcs)
    cs.sample_configuration(10)

    print('\nAll tests passed!')
