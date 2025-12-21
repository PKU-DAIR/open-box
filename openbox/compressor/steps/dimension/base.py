import copy
from typing import Optional, List, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from ...core.step import CompressionStep


class DimensionSelectionStep(CompressionStep):    
    def __init__(self, strategy: str = 'shap', **kwargs):
        super().__init__('dimension_selection', **kwargs)
        self.strategy = strategy
        self.selected_indices: Optional[List[int]] = None
        self.selected_param_names: Optional[List[str]] = None
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None,
                source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        if self.strategy == 'none':
            logger.debug("Dimension selection disabled, returning input space")
            return input_space
        selected_indices = self._select_parameters(input_space, space_history, source_similarities)
        if not selected_indices:
            logger.warning("No parameters selected, returning input space")
            return input_space
        
        compressed_space = self._create_compressed_space(input_space, selected_indices)
        self.selected_indices = selected_indices
        self.selected_param_names = [input_space.get_hyperparameter_names()[i] for i in selected_indices]
        logger.debug(f"Dimension selection: {len(input_space.get_hyperparameters())} -> "
                    f"{len(compressed_space.get_hyperparameters())} parameters")
        logger.debug(f"Selected parameters: {self.selected_param_names}")
        return compressed_space
    
    def _select_parameters(self, 
                          input_space: ConfigurationSpace,
                          space_history: Optional[List[History]] = None,
                          source_similarities: Optional[Dict[int, float]] = None) -> List[int]:
        """
        Select parameters to keep.
        
        Subclasses should override this method.
        
        Args:
            input_space: Input configuration space
            space_history: Historical data
            
        Returns:
            List of selected parameter indices
        """
        # Default: keep all parameters
        return list(range(len(input_space.get_hyperparameters())))
    
    def _create_compressed_space(self, 
                                 input_space: ConfigurationSpace,
                                 selected_indices: List[int]) -> ConfigurationSpace:
        param_names = input_space.get_hyperparameter_names()
        selected_names = [param_names[i] for i in selected_indices]
        
        compressed_space = ConfigurationSpace()
        for name in selected_names:
            hp = input_space.get_hyperparameter(name)
            compressed_space.add_hyperparameter(hp)
        
        return compressed_space
    
    def project_point(self, point) -> dict:
        # project a point from input_space to output_space.
        # filter to selected parameters and fill missing ones.
        if hasattr(point, 'get_dictionary'):
            point_dict = point.get_dictionary()
        elif isinstance(point, dict):
            point_dict = point
        else:
            point_dict = dict(point)
        
        # filter to only selected parameters
        if self.selected_param_names is None:
            filtered_dict = point_dict
        else:
            filtered_dict = {name: point_dict[name] for name in self.selected_param_names if name in point_dict}
        
        # fill missing parameters if needed
        if self.output_space is not None and self.filling_strategy is not None:
            filtered_dict = self.filling_strategy.fill_missing_parameters(
                filtered_dict, self.output_space
            )
        
        return filtered_dict
    
    def needs_unproject(self) -> bool:
        # Dimension selection is one-way, no unprojection needed
        return False
    
    def affects_sampling_space(self) -> bool:
        # Dimension selection affects sampling space
        return True
    
    def get_step_info(self) -> dict:
        info = super().get_step_info()
        if self.selected_param_names:
            info['selected_parameters'] = self.selected_param_names
        if self.selected_indices:
            info['selected_indices'] = self.selected_indices
        if hasattr(self, '_calculator') and self._calculator:
            info['calculator'] = type(self._calculator).__name__
        info['compression_ratio'] = len(self.selected_indices) / len(self.input_space.get_hyperparameters())
        return info