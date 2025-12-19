from typing import Optional, List, Dict
import numpy as np
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH

from ...core.step import CompressionStep


class TransformativeProjectionStep(CompressionStep):
    def __init__(self, method: str = 'rembo', **kwargs):
        super().__init__('transformative_projection', **kwargs)
        self.method = method
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None,
                source_similarities: Optional[Dict[int, float]] = None,
                **kwargs) -> ConfigurationSpace:
        if self.method == 'none':
            logger.info("Projection disabled, returning input space")
            return input_space
        
        projected_space = self._build_projected_space(input_space)
        
        logger.info(f"Projection compression: {len(input_space.get_hyperparameters())} -> "
                f"{len(projected_space.get_hyperparameters())} parameters")
        
        return projected_space
    
    def _build_projected_space(self, input_space: ConfigurationSpace) -> ConfigurationSpace:
        return input_space
    
    def needs_unproject(self) -> bool:
        return True
    
    def affects_sampling_space(self) -> bool:
        return True
    
    def _normalize_high_dim_config(self, high_dim_dict: dict, active_hps: List[CSH.Hyperparameter]) -> np.ndarray:
        high_dim_values = []
        for hp in active_hps:
            value = high_dim_dict.get(hp.name)
            if value is None:
                if hasattr(hp, 'default_value'):
                    value = hp.default_value
                elif hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                    value = (hp.lower + hp.upper) / 2
                elif hasattr(hp, 'choices'):
                    value = hp.choices[0]
                else:
                    logger.warning(f"Cannot determine value for {hp.name}, using 0.5")
                    high_dim_values.append(0.5)
                    continue
            # normalize to [0, 1]
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                normalized = (value - hp.lower) / (hp.upper - hp.lower)
            elif hasattr(hp, 'choices'):
                try:
                    normalized = hp.choices.index(value) / max(1, len(hp.choices) - 1)
                except ValueError:
                    normalized = 0.5
            else:
                normalized = 0.5
            high_dim_values.append(normalized)
        return np.array(high_dim_values)

