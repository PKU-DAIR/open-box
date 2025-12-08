import copy
from typing import Optional, List
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from ...core.step import CompressionStep


class RangeCompressionStep(CompressionStep):    
    def __init__(self, method: str = 'boundary', **kwargs):
        super().__init__('range_compression', **kwargs)
        self.method = method
        self.original_space: Optional[ConfigurationSpace] = None
    
    def compress(self, input_space: ConfigurationSpace, 
                 space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        if self.method == 'none':
            logger.info("Range compression disabled, returning input space")
            return input_space
        
        self.original_space = copy.deepcopy(input_space)
        compressed_space = self._compute_compressed_space(input_space, space_history)
        
        logger.info(f"Range compression: {len(input_space.get_hyperparameters())} parameters compressed")
        return compressed_space
    
    def _compute_compressed_space(self, 
                                  input_space: ConfigurationSpace,
                                  space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        return copy.deepcopy(input_space)
    
    def needs_unproject(self) -> bool:
        return self.method == 'quantization'
    
    def affects_sampling_space(self) -> bool:
        return True
