from abc import ABC, abstractmethod
from typing import Dict, Any
from ConfigSpace import ConfigurationSpace
from openbox import logger


class FillingStrategy(ABC):
    """
    Base class for strategies that fill missing parameters when converting
    configurations between spaces
    
    This is useful when:
    1. Small space -> Large space: Need to fill missing parameters
    2. Large space -> Small space: Already filtered, but may need to fill if space grows later
    
    All filling strategies support fixed_values parameter to override specific
    parameter values regardless of sampling or space ranges.
    """
    
    def __init__(self, fixed_values: Dict[str, Any] = None):
        self.fixed_values = fixed_values or {}
        if self.fixed_values:
            logger.debug(f"{self.__class__.__name__} initialized with {len(self.fixed_values)} fixed values: {list(self.fixed_values.keys())}")
    
    @abstractmethod
    def fill_missing_parameters(self, 
                                config_dict: Dict[str, Any],
                                target_space: ConfigurationSpace) -> Dict[str, Any]:
        pass
    
    def _apply_fixed_values(self, 
                            filled_dict: Dict[str, Any],
                            target_space: ConfigurationSpace) -> Dict[str, Any]:
        if not self.fixed_values:
            return filled_dict
        
        result_dict = filled_dict.copy()
        for param_name, fixed_value in self.fixed_values.items():
            if param_name in target_space.get_hyperparameter_names():
                original_value = result_dict.get(param_name)
                result_dict[param_name] = fixed_value
                if original_value != fixed_value:
                    logger.debug(f"Overrode parameter '{param_name}': {original_value} -> {fixed_value}")
            else:
                logger.warning(f"Fixed parameter '{param_name}' not found in target space")
        return result_dict
    
    def get_default_value(self, hp) -> Any:
        if hasattr(hp, 'default_value') and hp.default_value is not None:
            return hp.default_value
        
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            return (hp.lower + hp.upper) / 2.0
        elif hasattr(hp, 'choices') and len(hp.choices) > 0:
            return hp.choices[0]
        else:
            logger.warning(f"Cannot determine default value for {hp.name}, using None")
            return None

