from typing import Optional, List, Dict
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from openbox import logger

from .base import DimensionSelectionStep
from ...utils import load_expert_params


class ExpertDimensionStep(DimensionSelectionStep):
    def __init__(self, 
                 strategy: str = 'expert',
                 expert_params: Optional[List[str]] = None,
                 exclude_params: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(strategy=strategy, expert_params=expert_params, exclude_params=exclude_params, **kwargs)
    
    def get_step_info(self) -> dict:
        # Base class already includes expert_params info
        return super().get_step_info()
    
    def _select_parameters(self, 
                          input_space: ConfigurationSpace,
                          space_history: Optional[List[History]] = None,
                          source_similarities: Optional[Dict[int, float]] = None) -> List[int]:
        """
        Expert dimension selection: only use expert parameters.
        The base class will automatically include expert_params, so we return empty list
        to indicate no additional method-selected parameters.
        """
        if not self.expert_params:
            logger.warning("No expert parameters provided, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        # Return empty list - base class will handle expert params automatically
        # This ensures only expert parameters are selected
        return []

