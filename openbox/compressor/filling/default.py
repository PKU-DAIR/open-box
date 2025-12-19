from typing import Dict, Any
from ConfigSpace import ConfigurationSpace

from .base import FillingStrategy


class DefaultValueFilling(FillingStrategy):
    def fill_missing_parameters(self, 
                                config_dict: Dict[str, Any],
                                target_space: ConfigurationSpace) -> Dict[str, Any]:
        filled_dict = config_dict.copy()
        target_names = target_space.get_hyperparameter_names()
        
        for name in target_names:
            if name not in filled_dict:
                hp = target_space.get_hyperparameter(name)
                filled_dict[name] = self.get_default_value(hp)
        
        return filled_dict

