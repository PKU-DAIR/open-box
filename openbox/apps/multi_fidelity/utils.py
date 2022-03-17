# License: MIT

from typing import List
import numpy as np
from openbox.utils.config_space import Configuration, ConfigurationSpace

WAITING = 'waiting'
RUNNING = 'running'
COMPLETED = 'completed'
PROMOTED = 'promoted'


def sample_configuration(configuration_space: ConfigurationSpace, excluded_configs: List[Configuration] = None):
    """
    sample one config not in excluded_configs
    """
    if excluded_configs is None:
        excluded_configs = []
    sample_cnt = 0
    max_sample_cnt = 1000
    while True:
        config = configuration_space.sample_configuration()
        sample_cnt += 1
        if config not in excluded_configs:
            break
        if sample_cnt >= max_sample_cnt:
            raise ValueError('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
    return config


def sample_configurations(configuration_space: ConfigurationSpace, num: int,
                          excluded_configs: List[Configuration] = None) -> List[Configuration]:
    if excluded_configs is None:
        excluded_configs = []
    result = []
    cnt = 0
    while cnt < num:
        config = configuration_space.sample_configuration(1)
        if config not in result and config not in excluded_configs:
            result.append(config)
            cnt += 1
    return result


def expand_configurations(configs: List[Configuration], configuration_space: ConfigurationSpace, num: int,
                          excluded_configs: List[Configuration] = None):
    if excluded_configs is None:
        excluded_configs = []
    num_config = len(configs)
    num_needed = num - num_config
    config_cnt = 0
    while config_cnt < num_needed:
        config = configuration_space.sample_configuration(1)
        if config not in configs and config not in excluded_configs:
            configs.append(config)
            config_cnt += 1
    return configs


def minmax_normalization(x):
    min_value = min(x)
    delta = max(x) - min(x)
    if delta == 0:
        return [1.0] * len(x)
    return [(float(item) - min_value) / float(delta) for item in x]


def std_normalization(x):
    _mean = np.mean(x)
    _std = np.std(x)
    if _std == 0:
        return np.array([0.] * len(x))
    return (np.array(x) - _mean) / _std


def norm2_normalization(x):
    z = np.array(x)
    normalized_z = z / np.linalg.norm(z)
    return normalized_z
