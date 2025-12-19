from typing import Optional, List, Tuple
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .shap import SHAPDimensionStep
from ...core import OptimizerProgress


class PeriodicDimensionStep(SHAPDimensionStep):    
    def __init__(self, 
                 strategy: str = 'tuneful',
                 initial_topk: int = 50,
                 period: int = 10,
                 reduction_ratio: float = 0.4,
                 min_dimensions: int = 5,
                 source_similarities: Optional[List[Tuple[int, float]]] = None,
                 **kwargs):
        super().__init__(strategy='shap', topk=initial_topk, source_similarities=source_similarities, **kwargs)
        self.strategy = strategy
        self.current_topk = initial_topk
        self.initial_topk = initial_topk
        self.period = period
        self.reduction_ratio = reduction_ratio
        self.min_dimensions = min_dimensions

        self.original_space: Optional[ConfigurationSpace] = None
        self.space_history: Optional[List[History]] = None
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        self.original_space = input_space
        self.space_history = space_history
        
        if self.strategy == 'none':
            logger.debug("Tuneful dimension selection disabled, returning input space")
            return input_space
        
        self.topk = self.current_topk
        compressed_space = super().compress(input_space, space_history)
        
        logger.debug(f"Tuneful dimension selection (initial): {len(input_space.get_hyperparameters())} -> "
                    f"{len(compressed_space.get_hyperparameters())} parameters")
        
        return compressed_space
    
    def supports_adaptive_update(self) -> bool:
        return True
    
    def uses_progressive_compression(self) -> bool:
        # periodic uses progressive compression: compress on top of previous result
        return True
    
    def update(self, progress: OptimizerProgress, history: History) -> bool:
        if not progress.should_periodic_update(period=self.period):
            return False
        
        old_topk = self.current_topk
        reduction = int(self.current_topk * self.reduction_ratio)
        self.current_topk = max(self.min_dimensions, self.current_topk - reduction)
        
        if self.current_topk < old_topk:
            logger.debug(f"Tuneful periodic update (iteration {progress.iteration}): "
                    f"reducing dimensions {old_topk} -> {self.current_topk} "
                    f"(reduced by {reduction}, {self.reduction_ratio*100:.0f}%)")
            return True
        
        return False

