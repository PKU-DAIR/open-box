import json
import copy
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter
from openbox import space as sp, logger


def create_param(key, value):
    q_val = value.get('q', None)
    param_type = value['type']

    if param_type == 'integer':
        return sp.Int(key, value['min'], value['max'], default_value=value['default'], q=q_val)
    elif param_type == 'real':
        return sp.Real(key, value['min'], value['max'], default_value=value['default'], q=q_val)
    elif param_type == 'enum': 
        return sp.Categorical(key, value['enum_values'], default_value=value['default'])
    elif param_type == 'categorical':
        return sp.Categorical(key, value['choices'], default_value=value['default'])
    else:
        raise ValueError(f"Unsupported type: {param_type}")

def parse_combined_space(json_file_origin, json_file_new):
    if isinstance(json_file_origin, str):
        with open(json_file_origin, 'r') as f:
            conf = json.load(f)
        space = sp.Space()
        for key, value in conf.items():
            if key not in space.keys():
                para = create_param(key, value)
                space.add_variable(para)
    else:
        space = copy.deepcopy(json_file_origin)

    if isinstance(json_file_new, str):
        with open(json_file_new, 'r') as f:
            conf_new = json.load(f)
        for key, value in conf_new.items():
            if key not in space.keys():
                para = create_param(key, value)
                space.add_variable(para)
    else:
        for param in json_file_new.get_hyperparameters():
            if param.name not in space.keys():
                space.add_variable(param)
    
    return space

def create_space_from_ranges(
    original_space: ConfigurationSpace,
    compressed_ranges: Dict[str, Tuple[float, float]]
) -> ConfigurationSpace:
    compressed_space = copy.deepcopy(original_space)
    
    for param_name, (min_val, max_val) in compressed_ranges.items():
        try:
            hp = compressed_space.get_hyperparameter(param_name)
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                original_lower = hp.lower
                original_upper = hp.upper
                
                # Handle invalid range (min_val >= max_val)
                if min_val >= max_val:
                    if max_val < original_upper:
                        max_val = min_val + 1 \
                            if isinstance(hp, (sp.Int, UniformIntegerHyperparameter)) \
                            else max_val + (original_upper - original_lower) * 0.01
                        max_val = min(max_val, original_upper)
                    elif min_val > original_lower:
                        min_val = max_val - 1 \
                            if isinstance(hp, (sp.Int, UniformIntegerHyperparameter)) \
                                else max_val - (original_upper - original_lower) * 0.01
                        min_val = max(min_val, original_lower)
                    else:
                        min_val = original_lower
                        max_val = original_upper
                
                # Handle quantization
                q = hp.q if hasattr(hp, 'q') and hp.q is not None else None
                if q is not None:
                    min_val = np.floor(min_val / q) * q
                    max_val = np.ceil(max_val / q) * q
                    range_size = max_val - min_val
                    if range_size < q:
                        max_val = min_val + q
                    else:
                        range_size_rounded = np.round(range_size / q) * q
                        max_val = min_val + range_size_rounded
                
                if isinstance(hp, (sp.Int, UniformIntegerHyperparameter)) or isinstance(hp, sp.Int):
                    new_hp = UniformIntegerHyperparameter(
                        name=param_name,
                        lower=int(min_val),
                        upper=int(max_val),
                        default_value=int((min_val + max_val) / 2),
                        log=hp.log if hasattr(hp, 'log') else False,
                        q=q
                    )
                elif isinstance(hp, (sp.Real, UniformFloatHyperparameter)) or isinstance(hp, sp.Real):
                    new_hp = UniformFloatHyperparameter(
                        name=param_name,
                        lower=float(min_val),
                        upper=float(max_val),
                        default_value=float((min_val + max_val) / 2),
                        q=q,
                        log=hp.log if hasattr(hp, 'log') else False
                    )
                else:
                    logger.warning(f"Unsupported hyperparameter type for {param_name}: {type(hp)}")
                    continue
                compressed_space._hyperparameters.pop(param_name)
                compressed_space.add_hyperparameter(new_hp)
                
                logger.info(
                    f"Compressed {param_name}: [{original_lower}, {original_upper}] -> "
                    f"[{new_hp.lower}, {new_hp.upper}]"
                )
        except Exception as e:
            logger.warning(f"Failed to compress parameter {param_name}: {e}")
            
    return compressed_space


def load_performance_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logger.debug(f"Loaded {len(data)} records from {data_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        return None

def extract_top_samples_from_history(
    space_history: List[History],
    numeric_param_names: List[str],
    input_space: ConfigurationSpace,
    top_ratio: float = 1.0,
    normalize: bool = True,
    return_history_indices: bool = False
) -> Union[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray], List[int]]]:
    all_x = []
    all_y = []
    history_indices = [] if return_history_indices else None
    
    for task_idx, history in enumerate(space_history):
        if len(history) == 0:
            continue
        
        valid_configs = []
        valid_objectives = []
        
        for obs in history.observations:
            if obs.objectives and len(obs.objectives) > 0:
                obj_value = obs.objectives[0]
                if np.isfinite(obj_value):
                    valid_configs.append(obs.config)
                    valid_objectives.append(obj_value)
        
        if len(valid_configs) == 0:
            logger.debug(f"Skipping history with no valid objectives")
            continue
        
        x_numeric = extract_numeric_values_from_configs(
            valid_configs, numeric_param_names, input_space, normalize=normalize
        )
        
        objectives_array = np.array(valid_objectives)
        
        if top_ratio < 1.0:
            sorted_indices = np.argsort(objectives_array)
            top_n = max(1, int(len(sorted_indices) * top_ratio))
            top_indices = sorted_indices[: top_n]
            all_x.append(x_numeric[top_indices])
            all_y.append(objectives_array[top_indices])
            if return_history_indices:
                history_indices.extend([task_idx] * len(top_indices))
        else:
            all_x.append(x_numeric)
            all_y.append(objectives_array)
            if return_history_indices:
                history_indices.extend([task_idx] * len(x_numeric))
    
    return (all_x, all_y, history_indices) if return_history_indices else (all_x, all_y)


def extract_numeric_values_from_configs(
    configs: List[Union[Configuration, Dict]],
    numeric_param_names: List[str],
    input_space: ConfigurationSpace,
    normalize: bool = True
) -> np.ndarray:
    n_samples = len(configs)
    n_params = len(numeric_param_names)
    X = np.zeros((n_samples, n_params))
    
    for i, param_name in enumerate(numeric_param_names):
        try:
            hp = input_space.get_hyperparameter(param_name)
        except KeyError:
            logger.warning(f"Parameter {param_name} not found in input_space, skipping")
            continue
        
        if not hasattr(hp, 'lower') or not hasattr(hp, 'upper'):
            logger.warning(f"Parameter {param_name} is not numeric, skipping")
            continue
        
        lower = hp.lower
        upper = hp.upper
        range_size = upper - lower
        
        for j, config in enumerate(configs):
            value = None
            if hasattr(config, 'get'):
                value = config.get(param_name)
            elif isinstance(config, dict):
                value = config.get(param_name)
            elif hasattr(config, param_name):
                value = getattr(config, param_name, None)
            
            if value is None:
                value = hp.default_value
                logger.warning(f"Parameter {param_name} not found in config {j}, using default {value}")
            
            if normalize and range_size > 0:
                normalized_value = (value - lower) / range_size
                X[j, i] = normalized_value
            else:
                X[j, i] = value
    return X


def load_expert_params(expert_config_file: str, key: str = 'spark') -> List[str]:
    try:
        with open(expert_config_file, "r") as f:
            all_expert_params = json.load(f)   
        expert_params = all_expert_params.get(key, [])             
        return expert_params
        
    except FileNotFoundError:
        logger.warning(f"Expert config file not found: {expert_config_file}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing expert config file: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading expert parameters: {e}")
        return []


def collect_compression_details(original_space: ConfigurationSpace, compressed_space: ConfigurationSpace) -> Dict[str, Any]:
    details = {}
    range_hp_names = [hp.name for hp in compressed_space.get_hyperparameters()]
    
    for hp in original_space.get_hyperparameters():
        name = hp.name
        if name in range_hp_names:
            original_hp = hp
            compressed_hp = compressed_space.get_hyperparameter(name)
            
            if hasattr(original_hp, 'lower') and hasattr(original_hp, 'upper'): # Numeric hyperparameter
                details[name] = {
                    'type': 'numeric',
                    'original_range': [original_hp.lower, original_hp.upper],
                    'compressed_range': [compressed_hp.lower, compressed_hp.upper],
                    'original_default': original_hp.default_value,
                    'compressed_default': compressed_hp.default_value,
                    'compression_ratio': (compressed_hp.upper - compressed_hp.lower) / (original_hp.upper - original_hp.lower)
                }
            elif hasattr(original_hp, 'choices'):   # Categorical hyperparameter
                details[name] = {
                    'type': 'categorical',
                    'original_choices': list(original_hp.choices),
                    'compressed_choices': list(compressed_hp.choices),
                    'original_default': original_hp.default_value,
                    'compressed_default': compressed_hp.default_value,
                    'compression_ratio': len(compressed_hp.choices) / len(original_hp.choices)
                }
    return details


def extract_numeric_hyperparameters(space: ConfigurationSpace) -> Tuple[List[str], List[int]]:
    numeric_hyperparameter_indices = []
    numeric_hyperparameter_names = []
    for i, hp in enumerate(space.get_hyperparameters()):
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            numeric_hyperparameter_names.append(hp.name)
            numeric_hyperparameter_indices.append(i)
    return numeric_hyperparameter_names, numeric_hyperparameter_indices