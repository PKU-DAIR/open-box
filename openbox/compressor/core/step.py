"""
Compression step base class and interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, TYPE_CHECKING
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
    
    @abstractmethod
    def compress(self, input_space: ConfigurationSpace, 
                 space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        pass
    
    def project_point(self, point) -> dict:
        if hasattr(point, 'get_dictionary'):
            return point.get_dictionary()
        elif isinstance(point, dict):
            return point
        else:
            return dict(point)
    
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
