from typing import Optional, List, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .base import DimensionSelectionStep
from ...utils import load_expert_params


class ExpertDimensionStep(DimensionSelectionStep):
    def __init__(self, 
                 strategy: str = 'expert',
                 expert_params: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.expert_params = expert_params
    
    def _select_parameters(self, 
                          input_space: ConfigurationSpace,
                          space_history: Optional[List[History]] = None,
                          source_similarities: Optional[Dict[int, float]] = None) -> List[int]:
        if not self.expert_params:
            logger.warning("No expert parameters provided, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
                
        param_names = input_space.get_hyperparameter_names()
        selected_indices = []
        valid_params = []
        
        for param_name in self.expert_params:
            if param_name in param_names:
                idx = param_names.index(param_name)
                selected_indices.append(idx)
                valid_params.append(param_name)
            else:
                logger.warning(f"Expert parameter '{param_name}' not found in configuration space")
        
        if not selected_indices:
            logger.warning("No valid expert parameters found, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        logger.debug(f"Expert dimension selection: {len(selected_indices)} parameters selected")
        logger.debug(f"Selected parameters: {valid_params}")
        
        return sorted(selected_indices)

