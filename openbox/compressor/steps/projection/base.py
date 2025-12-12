from typing import Optional, List
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from ...core.step import CompressionStep


class TransformativeProjectionStep(CompressionStep):
    def __init__(self, method: str = 'rembo', **kwargs):
        super().__init__('transformative_projection', **kwargs)
        self.method = method
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        # space_history is not used for projection
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