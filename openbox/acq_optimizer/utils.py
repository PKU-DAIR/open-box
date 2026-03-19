import numpy as np
from typing import List
from ConfigSpace import Configuration, ConfigurationSpace


def convert_configurations_to_array(configs: List[Configuration]) -> np.ndarray:
    if not configs:
        raise ValueError("configs list cannot be empty")
    
    configs_array = np.array(
        [config.get_array() for config in configs],
        dtype=np.float64
    )
    configuration_space = configs[0].configuration_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
    configuration_space: ConfigurationSpace,
    configs_array: np.ndarray
) -> np.ndarray:
    for hp in configuration_space.get_hyperparameters():
        default = hp.normalized_default_value
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)
        
        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = default
    
    return configs_array