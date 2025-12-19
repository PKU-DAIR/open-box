import numpy as np
from typing import Optional, List, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .base import DimensionSelectionStep
from .importance import SHAPImportanceCalculator


class SHAPDimensionStep(DimensionSelectionStep):
    
    def __init__(self, 
                 strategy: str = 'shap', 
                 topk: int = 20,
                 **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.topk = 0 if strategy == 'none' else topk
        self._calculator = SHAPImportanceCalculator()
        logger.debug(f"SHAPDimensionStep initialized: topk={topk}")
    
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
        
        top_k = min(self.topk, len(param_names))
        if top_k == 0:
            logger.warning("No numeric hyperparameters detected, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        selected_numeric_indices = np.argsort(importances)[: top_k].tolist()
        selected_param_names = [param_names[i] for i in selected_numeric_indices]
        importances_selected = importances[selected_numeric_indices]
        
        all_param_names = input_space.get_hyperparameter_names()
        selected_indices = [all_param_names.index(name) for name in selected_param_names]
        
        logger.debug(f"SHAP dimension selection: {selected_param_names}")
        logger.debug(f"SHAP importances: {importances_selected}")
        
        return selected_indices
