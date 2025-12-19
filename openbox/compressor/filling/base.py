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
    """
    
    @abstractmethod
    def fill_missing_parameters(self, 
                                config_dict: Dict[str, Any],
                                target_space: ConfigurationSpace) -> Dict[str, Any]:
        pass
    
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

