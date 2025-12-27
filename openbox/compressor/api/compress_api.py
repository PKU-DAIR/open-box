"""
Compression API for Frontend Integration

This module provides a high-level API for frontend applications to:
1. Create compression steps from string identifiers
2. Execute compression with user-provided configuration
3. Return results in JSON format
"""

import json
import sys
import argparse
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from openbox.utils.history import History, Observation
from openbox.utils.constants import SUCCESS

from .. import Compressor
from .step_factory import (
    validate_step_string,
    create_steps_from_strings,
)
from .filling_factory import (
    create_filling_from_string,
    create_filling_from_config,
)
from openbox import logger


def create_config_space_from_dict(config_dict: Dict[str, Any]) -> ConfigurationSpace:
    """
    Create a ConfigurationSpace from a dictionary definition.
    
    Args:
        config_dict: Dictionary with hyperparameter definitions.
                     Format: {
                         'param_name': {
                             'type': 'float' | 'integer' | 'int' | 'categorical',
                             'min': float,  # for float/integer/int
                             'max': float,  # for float/integer/int
                             'default': value,
                             'log': bool,  # optional, for float/integer/int
                             'choices': List,  # for categorical
                         }
                     }
    
    Returns:
        ConfigurationSpace instance
    """
    cs = ConfigurationSpace()
    
    for param_name, param_def in config_dict.items():
        param_type = param_def.get('type', 'float').lower()
        
        if param_type in ('float', 'real'):
            hp = UniformFloatHyperparameter(
                name=param_name,
                lower=float(param_def['min']),
                upper=float(param_def['max']),
                default_value=param_def.get('default', (param_def['min'] + param_def['max']) / 2),
                log=param_def.get('log', False)
            )
        elif param_type in ('int', 'integer'):
            hp = UniformIntegerHyperparameter(
                name=param_name,
                lower=int(param_def['min']),
                upper=int(param_def['max']),
                default_value=param_def.get('default', int((param_def['min'] + param_def['max']) / 2)),
                log=param_def.get('log', False)
            )
        elif param_type == 'categorical':
            hp = CategoricalHyperparameter(
                name=param_name,
                choices=param_def['choices'],
                default_value=param_def.get('default', param_def['choices'][0])
            )
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}. Supported: 'float', 'integer', 'int', 'categorical'")
        
        cs.add_hyperparameter(hp)
    
    return cs


def load_history_from_dict(
    history_data: List[Dict[str, Any]],
    config_space: ConfigurationSpace
) -> History:
    """
    Create a History object from dictionary data.
    
    Supports two formats:
    1. Simple format:
       {
           'config': Dict[str, Any],
           'objective': float,  # single value
           'trial_state': str,  # optional
           'elapsed_time': float  # optional
       }
    
    2. Full format (from History Type):
       {
           'config': Dict[str, Any],
           'objectives': List[float],  # array of objectives
           'constraints': List[float] | None,  # optional
           'trial_state': int | str,  # 0 or 'SUCCESS'
           'elapsed_time': float,  # optional
           'create_time': str,  # optional, ignored
           'extra_info': dict  # optional, ignored
       }
    
    Args:
        history_data: List of observations
        config_space: ConfigurationSpace instance
    
    Returns:
        History instance
    """
    num_objectives = 1
    num_constraints = 0
    if history_data:
        first_obs = history_data[0]
        if 'objectives' in first_obs:
            num_objectives = len(first_obs['objectives'])
        if 'constraints' in first_obs and first_obs['constraints'] is not None:
            num_constraints = len(first_obs['constraints'])
    
    history = History(
        task_id='frontend_task',
        num_objectives=num_objectives,
        num_constraints=num_constraints,
        config_space=config_space
    )
    
    for obs_data in history_data:
        config = Configuration(config_space, values=obs_data['config'])
        
        if 'objectives' in obs_data:
            objectives = obs_data['objectives']
        elif 'objective' in obs_data:
            objectives = [obs_data['objective']]
        else:
            raise ValueError("Observation must have either 'objective' or 'objectives' field")
        
        constraints = obs_data.get('constraints', None)
        
        trial_state = obs_data.get('trial_state', SUCCESS)
        if isinstance(trial_state, int):
            if trial_state == 0:
                trial_state = SUCCESS
        
        obs = Observation(
            config=config,
            objectives=objectives,
            constraints=constraints,
            trial_state=trial_state,
            elapsed_time=obs_data.get('elapsed_time', 0.0)
        )
        history.update_observation(obs)
    
    return history


def load_histories_from_dicts(
    histories_data: List[List[Dict[str, Any]]],
    config_space: ConfigurationSpace
) -> List[History]:
    """
    Load multiple History objects from list of observation dictionaries.
    
    Args:
        histories_data: List of history data, each is a list of observations.
                       Format: [
                           [obs1, obs2, ...],  # History 1
                           [obs3, obs4, ...],  # History 2
                           ...
                       ]
        config_space: ConfigurationSpace instance
    
    Returns:
        List of History instances
    """
    histories = []
    for i, history_data in enumerate(histories_data):
        history = load_history_from_dict(history_data, config_space)
        history.task_id = f'source_task_{i}'
        histories.append(history)
        logger.info(f"Loaded history {i+1}: {len(history.observations)} observations")
    return histories


def compress_from_config(
    config_space_def: Dict[str, Any],
    step_config: Dict[str, Any],
    history_data: List[List[Dict[str, Any]]],
    output_dir: Optional[str] = None,
    save_info: bool = True
) -> Dict[str, Any]:
    """
    Execute compression from configuration dictionary.
    
    Args:
        config_space_def: Configuration space definition (see create_config_space_from_dict)
        step_config: Step configuration dictionary with format:
                    {
                        'dimension_step': str,  # e.g., 'd_shap', 'd_none'
                        'range_step': str,      # e.g., 'r_kde', 'r_none'
                        'projection_step': str, # e.g., 'p_quant', 'p_none'
                        'step_params': {        # optional, parameter overrides
                            'd_shap': {'topk': 10, 'exclude_params': ['param1']},
                            'r_kde': {'source_top_ratio': 0.5}
                        },
                        'filling_config': {     # optional, filling strategy config
                            'type': 'default',
                            'fixed_values': {   # optional, fixed parameter values
                                'param1': value1,
                                'param2': value2
                            }
                        }
                    }
        history_data: Required history data. List[List[Dict]] format:
                     - Single history: [[...]] (one JSON file with list of observations)
                     - Multiple histories: [[...], [...], ...] (multiple JSON files, each with list of observations)
                     Each inner list (JSON file) will be converted to one History object.
                     Source similarities will be automatically set to 1/len(histories) for each history.
        output_dir: Optional output directory for saving results
        save_info: Whether to save compression info
    
    Returns:
        Dictionary with compression results
    """
    config_space = create_config_space_from_dict(config_space_def)
    logger.info(f"Created configuration space with {len(config_space.get_hyperparameters())} parameters")
    

    step_strings = []
    step_params = step_config.get('step_params', {})
    
    dim_step = step_config.get('dimension_step', 'd_none')
    if validate_step_string(dim_step):
        step_strings.append(dim_step)
    else:
        raise ValueError(f"Invalid dimension step: {dim_step}")
    
    range_step = step_config.get('range_step', 'r_none')
    if validate_step_string(range_step):
        step_strings.append(range_step)
    else:
        raise ValueError(f"Invalid range step: {range_step}")
    
    proj_step = step_config.get('projection_step', 'p_none')
    if validate_step_string(proj_step):
        step_strings.append(proj_step)
    else:
        raise ValueError(f"Invalid projection step: {proj_step}")
    
    steps = create_steps_from_strings(step_strings, step_params=step_params)
    logger.info(f"Created {len(steps)} compression steps")
    
    # create filling strategy from config if provided
    filling_strategy = None
    filling_config = step_config.get('filling_config')
    if filling_config:
        filling_strategy = create_filling_from_config(filling_config)
        logger.info(f"Created filling strategy: {type(filling_strategy).__name__}")
        if filling_strategy.fixed_values:
            logger.info(f"Fixed values: {list(filling_strategy.fixed_values.keys())}")

    compressor = Compressor(
        config_space=config_space,
        steps=steps,
        filling_strategy=filling_strategy,
        save_compression_info=save_info,
        output_dir=output_dir
    )
    
    space_history = []
    for i, history_dict_list in enumerate(history_data):
        history = load_history_from_dict(history_dict_list, config_space)
        history.task_id = f'source_task_{i}'
        space_history.append(history)
        logger.info(f"Loaded history {i+1}: {len(history.observations)} observations")
    
    num_histories = len(space_history)
    source_similarities = {i: 1.0 / num_histories for i in range(num_histories)}
    logger.info(f"Loaded {len(space_history)} histories with auto-calculated similarities: {source_similarities}")
    
    surrogate_space, sample_space = compressor.compress_space(
        space_history=space_history,
        source_similarities=source_similarities
    )
    
    result = {
        'success': True,
        'original_dim': len(config_space.get_hyperparameters()),
        'surrogate_dim': len(surrogate_space.get_hyperparameters()),
        'sample_dim': len(sample_space.get_hyperparameters()),
        'compression_ratio': len(surrogate_space.get_hyperparameters()) / len(config_space.get_hyperparameters()),
        'original_params': config_space.get_hyperparameter_names(),
        'surrogate_params': surrogate_space.get_hyperparameter_names(),
        'sample_params': sample_space.get_hyperparameter_names(),
        'steps_used': [type(s).__name__ for s in steps],
        'output_dir': output_dir if save_info else None
    }
    
    try:
        summary = compressor.get_compression_summary()
        result['compression_summary'] = summary
    except:
        pass
    logger.info(f"Compression completed: {result['original_dim']} -> {result['surrogate_dim']} dimensions")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Compression API - Execute compression from JSON configuration'
    )
    parser.add_argument(
        '--config-space',
        type=str,
        required=True,
        help='Path to JSON file with configuration space definition'
    )
    parser.add_argument(
        '--steps',
        type=str,
        required=True,
        help='Path to JSON file with step configuration'
    )
    parser.add_argument(
        '--history',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to JSON file(s) with history data (required). Can specify multiple files for multi-source transfer learning. Source similarities will be auto-calculated as 1/len(histories) for each history.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for compression results (default: ./results/compression)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save compression info'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    config_space_path = Path(args.config_space)
    if not config_space_path.exists():
        print(f"Error: Config space file not found: {args.config_space}", file=sys.stderr)
        sys.exit(1)
    with open(config_space_path, 'r') as f:
        config_space_def = json.load(f)
    
    steps_path = Path(args.steps)
    if not steps_path.exists():
        print(f"Error: Steps config file not found: {args.steps}", file=sys.stderr)
        sys.exit(1)
    with open(steps_path, 'r') as f:
        step_config = json.load(f)
    
    histories_list = []
    for hist_file in args.history:
        hist_path = Path(hist_file)
        if not hist_path.exists():
            print(f"Error: History file not found: {hist_file}", file=sys.stderr)
            sys.exit(1)
        with open(hist_path, 'r') as f:
            hist_data = json.load(f)
            if isinstance(hist_data, dict) and 'observations' in hist_data:
                hist_data = hist_data['observations']
            histories_list.append(hist_data)
    history_data = histories_list
    
    try:
        result = compress_from_config(
            config_space_def=config_space_def,
            step_config=step_config,
            history_data=history_data,
            output_dir=args.output_dir,
            save_info=not args.no_save
        )
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

