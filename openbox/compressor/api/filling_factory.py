"""
Filling Factory Module
Provides functions to create filling strategy instances from string identifiers or configurations.
"""

from typing import Optional, Dict, Any
from ..filling import FillingStrategy, DefaultValueFilling
from openbox import logger

_FILLING_REGISTRY = {
    'default': {
        'class': DefaultValueFilling,
        'default_params': {},
        'description': 'Default value filling strategy',
    },
}


def get_available_filling_strings() -> list:
    return list(_FILLING_REGISTRY.keys())

def validate_filling_string(filling_str: str) -> bool:
    return filling_str in _FILLING_REGISTRY

def get_filling_info(filling_str: str) -> Optional[Dict[str, Any]]:
    if not validate_filling_string(filling_str):
        return None
    
    info = _FILLING_REGISTRY[filling_str].copy()
    if info['class'] is not None:
        info['class_name'] = info['class'].__name__
    else:
        info['class_name'] = None
    info.pop('class', None)
    return info

def create_filling_from_string(
    filling_str: str = 'default',
    fixed_values: Optional[Dict[str, Any]] = None,
    **kwargs
) -> FillingStrategy:
    """
    Create a filling strategy instance from a string identifier.
    
    Args:
        filling_str: Filling strategy string identifier (default: 'default')
        fixed_values: Optional dictionary mapping parameter names to their fixed values.
                     Example: {'learning_rate': 0.001, 'batch_size': 32}
        **kwargs: Additional parameters for the filling strategy
        
    Returns:
        FillingStrategy instance
        
    Examples:
        >>> # Create default filling strategy
        >>> filling = create_filling_from_string('default')
        
        >>> # Create filling strategy with fixed values
        >>> filling = create_filling_from_string(
        ...     'default',
        ...     fixed_values={'learning_rate': 0.001, 'batch_size': 32}
        ... )
    """
    if not validate_filling_string(filling_str):
        logger.error(f"Invalid filling strategy string identifier: {filling_str}")
        logger.info(f"Available filling strings: {list(_FILLING_REGISTRY.keys())}")
        raise ValueError(
            f"Invalid filling strategy string identifier: {filling_str}. "
            f"Available options: {list(_FILLING_REGISTRY.keys())}"
        )
    
    registry_entry = _FILLING_REGISTRY[filling_str]
    filling_class = registry_entry['class']
    
    if filling_class is None:
        logger.error(f"Filling strategy '{filling_str}' maps to None")
        raise ValueError(f"Filling strategy '{filling_str}' is not available")
    
    default_params = registry_entry['default_params'].copy()
    default_params.update(kwargs)
    if fixed_values is not None:
        default_params['fixed_values'] = fixed_values
    
    try:
        filling = filling_class(**default_params)
        logger.debug(
            f"Created {filling_class.__name__} from '{filling_str}' "
            f"with parameters: {default_params}"
        )
        return filling
    except Exception as e:
        logger.error(
            f"Failed to create filling strategy from '{filling_str}': {e}. "
            f"Parameters: {default_params}"
        )
        raise


def create_filling_from_config(
    config: Optional[Dict[str, Any]] = None
) -> FillingStrategy:
    """
    Create a filling strategy from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with the following structure:
                {
                    'type': 'default',  # filling strategy type
                    'fixed_values': {   # optional fixed values
                        'param1': value1,
                        'param2': value2,
                    }
                }
                If None, returns default filling strategy.
        
    Returns:
        FillingStrategy instance
        
    Examples:
        >>> # Create from config with fixed values
        >>> config = {
        ...     'type': 'default',
        ...     'fixed_values': {'learning_rate': 0.001}
        ... }
        >>> filling = create_filling_from_config(config)
    """
    if config is None:
        return create_filling_from_string('default')
    
    filling_str = config.get('type', 'default')
    fixed_values = config.get('fixed_values', None)
    
    return create_filling_from_string(
        filling_str=filling_str,
        fixed_values=fixed_values
    )

