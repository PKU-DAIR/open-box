import numpy as np
from typing import Optional, List, Dict
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from openbox import logger

from .base import DimensionSelectionStep
from .importance import SHAPImportanceCalculator


class SHAPDimensionStep(DimensionSelectionStep):
    
    def __init__(self, 
                 strategy: str = 'shap', 
                 topk: int = 20,
                 expert_params: Optional[List[str]] = None,
                 exclude_params: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(strategy=strategy, expert_params=expert_params, exclude_params=exclude_params, **kwargs)
        self.topk = 0 if strategy == 'none' else topk
        self._calculator = SHAPImportanceCalculator()
        logger.debug(f"SHAPDimensionStep initialized: topk={topk}, expert_params={len(self.expert_params)}, exclude_params={len(self.exclude_params)}")
    
    def get_step_info(self) -> dict:
        info = super().get_step_info()
        info['topk'] = self.topk
        return info
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None,
                source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        return super().compress(input_space, space_history, source_similarities)
    
    def _select_parameters(self, 
                        input_space: ConfigurationSpace,
                        space_history: Optional[List[History]] = None,
                        source_similarities: Optional[Dict[int, float]] = None) -> List[int]:
        if self.topk <= 0:
            logger.warning("No topk provided for SHAP selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        if not space_history:
            logger.warning("No space history provided for SHAP selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        param_names, importances = self._calculator.calculate_importances(
            input_space, space_history, source_similarities
        )
        
        if importances is None or np.size(importances) == 0:
            logger.warning("SHAP importances unavailable, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        # Return all parameters sorted by importance (not just topk)
        # Base class will select topk from this sorted list
        sorted_numeric_indices = np.argsort(importances).tolist()
        # sorted_numeric_indices.reverse()
        all_param_names = input_space.get_hyperparameter_names()
        sorted_indices = [all_param_names.index(param_names[i]) for i in sorted_numeric_indices]
        
        top_k = min(self.topk, len(sorted_indices))
        topk_indices = sorted_indices[:top_k] if top_k > 0 else []
        topk_names = [all_param_names[i] for i in topk_indices]
        topk_importances = importances[sorted_numeric_indices[:top_k]] if top_k > 0 else []
        
        logger.debug(f"SHAP sorted all {len(sorted_indices)} parameters by importance")
        logger.debug(f"SHAP top-{top_k} parameters: {topk_names}")
        logger.debug(f"SHAP top-{top_k} importances: {topk_importances}")        
        return sorted_indices
