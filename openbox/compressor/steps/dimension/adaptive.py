import numpy as np
from typing import Optional, List, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .base import DimensionSelectionStep
from .importance import ImportanceCalculator, SHAPImportanceCalculator
from ...core import OptimizerProgress
from ...core.update import UpdateStrategy, PeriodicUpdateStrategy


class AdaptiveDimensionStep(DimensionSelectionStep):    
    def __init__(self,
                 importance_calculator: Optional[ImportanceCalculator] = None,
                 update_strategy: Optional[UpdateStrategy] = None,
                 initial_topk: int = 30,
                 reduction_ratio: float = 0.2,
                 min_dimensions: int = 5,
                 max_dimensions: Optional[int] = None,
                 **kwargs):
        super().__init__(strategy='adaptive', **kwargs)
        
        self.importance_calculator = importance_calculator or SHAPImportanceCalculator()
        self.update_strategy = update_strategy or PeriodicUpdateStrategy(period=5)
        
        self.current_topk = initial_topk
        self.initial_topk = initial_topk
        self.reduction_ratio = reduction_ratio
        self.min_dimensions = min_dimensions
        self.max_dimensions = max_dimensions
        
        self.original_space: Optional[ConfigurationSpace] = None
        self.space_history: Optional[List[History]] = None
        
        if self.update_strategy:
            logger.info(f"AdaptiveDimensionStep initialized: "
                       f"importance={self.importance_calculator.get_name()}, "
                       f"update={self.update_strategy.get_name()}, "
                       f"initial_topk={initial_topk}")
        else:
            logger.info(f"AdaptiveDimensionStep initialized: "
                       f"importance={self.importance_calculator.get_name()}, "
                       f"update=None (fixed topk={initial_topk})")
    
    def compress(self, input_space: ConfigurationSpace,
                space_history: Optional[List[History]] = None,
                source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        self.original_space = input_space
        self.space_history = space_history
        
        param_names, importances = self.importance_calculator.calculate_importances(
            input_space, space_history, source_similarities
        )
        
        if len(param_names) == 0:
            logger.warning("No numeric parameters detected, returning input space")
            return input_space
        
        topk = min(self.current_topk, len(param_names))
        if self.max_dimensions is not None:
            topk = min(topk, self.max_dimensions)
        topk = max(topk, self.min_dimensions)
        
        selected_numeric_indices = np.argsort(importances)[:topk].tolist()
        selected_param_names = [param_names[i] for i in selected_numeric_indices]
        importances_selected = importances[selected_numeric_indices]
        
        logger.debug(f"{self.importance_calculator.get_name()} selected parameters: {selected_param_names}")
        logger.debug(f"{self.importance_calculator.get_name()} importances: {importances_selected}")
        
        all_param_names = input_space.get_hyperparameter_names()
        selected_indices = [all_param_names.index(name) for name in selected_param_names]
        
        compressed_space = self._create_compressed_space(input_space, selected_indices)
        self.selected_indices = selected_indices
        self.selected_param_names = selected_param_names
        
        logger.debug(f"Adaptive dimension selection: {len(input_space.get_hyperparameters())} -> "
                    f"{len(compressed_space.get_hyperparameters())} parameters "
                    f"(using {self.importance_calculator.get_name()})")
        
        return compressed_space
    
    def supports_adaptive_update(self) -> bool:
        return self.update_strategy is not None
    
    def uses_progressive_compression(self) -> bool:
        return True
    
    def update(self, progress: OptimizerProgress, history: History) -> bool:
        if not self.update_strategy:
            return False
        
        if not self.update_strategy.should_update(progress, history):
            return False
        
        old_topk = self.current_topk
        
        new_topk, description = self.update_strategy.compute_new_topk(
            current_topk=self.current_topk,
            reduction_ratio=self.reduction_ratio,
            min_dimensions=self.min_dimensions,
            max_dimensions=self.max_dimensions,
            progress=progress
        )
        
        if new_topk != old_topk:
            self.current_topk = new_topk
            logger.info(description)
            return True
        
        return False
    
    def get_step_info(self) -> dict:
        info = super().get_step_info()
        info['importance_calculator'] = type(self.importance_calculator).__name__

        if self.update_strategy:
            info['update_strategy'] = self.update_strategy.get_name()

        info['current_topk'] = self.current_topk
        info['initial_topk'] = self.initial_topk
        info['min_dimensions'] = self.min_dimensions
        info['reduction_ratio'] = self.reduction_ratio
        if self.max_dimensions:
            info['max_dimensions'] = self.max_dimensions
        return info
