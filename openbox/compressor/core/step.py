"""
Compression step base class and interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, TYPE_CHECKING
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .progress import OptimizerProgress
if TYPE_CHECKING:
    from ..sampling import SamplingStrategy


class CompressionStep(ABC):    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.input_space: Optional[ConfigurationSpace] = None
        self.output_space: Optional[ConfigurationSpace] = None
        self.filling_strategy = None  # Will be set by pipeline/compressor
    
    @abstractmethod
    def compress(self, input_space: ConfigurationSpace, 
                 space_history: Optional[List[History]] = None,
                 source_similarities: Optional[Dict[int, float]] = None,
                 **kwargs) -> ConfigurationSpace:
        pass
    
    def project_point(self, point) -> dict:
        # project a point from input_space to output_space.
        # fill missing parameters if output_space is set
        if hasattr(point, 'get_dictionary'):
            point_dict = point.get_dictionary()
        elif isinstance(point, dict):
            point_dict = point
        else:
            point_dict = dict(point)
        
        if self.output_space is not None and self.filling_strategy is not None:
            point_dict = self.filling_strategy.fill_missing_parameters(
                point_dict, self.output_space
            )
        
        return point_dict
    
    def unproject_point(self, point) -> dict:
        if hasattr(point, 'get_dictionary'):
            return point.get_dictionary()
        elif isinstance(point, dict):
            return point
        else:
            return dict(point)
    
    def needs_unproject(self) -> bool:
        return False
    
    def affects_sampling_space(self) -> bool:
        return False
    
    def update(self, progress: 'OptimizerProgress', history: History) -> bool:
        # True if compression was updated and needs re-compression
        return False
    
    def get_sampling_strategy(self) -> Optional['SamplingStrategy']:
        return None
    
    def supports_adaptive_update(self) -> bool:
        return False
    
    def uses_progressive_compression(self) -> bool:
        """
        Whether this step uses progressive compression (compress on top of previous compression)
        or re-compression (compress from original space).
        
        Progressive: periodic dimension reduction (30d -> 24d -> 19d)
        Re-compression: re-evaluate from scratch based on new data
        """
        return False
    
    def get_step_info(self) -> dict:
        info = {
            'name': self.name,
            'type': type(self).__name__,
            'input_space_params': len(self.input_space.get_hyperparameters()) if self.input_space else 0,
            'output_space_params': len(self.output_space.get_hyperparameters()) if self.output_space else 0,
            'supports_adaptive_update': self.supports_adaptive_update(),
            'uses_progressive_compression': self.uses_progressive_compression()
        }
        
        if hasattr(self, 'compression_info') and self.compression_info:
            info['compression_info'] = self.compression_info
        return info

