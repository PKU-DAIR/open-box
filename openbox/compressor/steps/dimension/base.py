import copy
from typing import Optional, List, Dict
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from openbox import logger

from ...core.step import CompressionStep


class DimensionSelectionStep(CompressionStep):    
    def __init__(self, strategy: str = 'shap', 
                 expert_params: Optional[List[str]] = None,
                 exclude_params: Optional[List[str]] = None, 
                 **kwargs):
        super().__init__('dimension_selection', **kwargs)
        self.strategy = strategy
        self.expert_params = expert_params or []
        self.exclude_params = exclude_params or []
        self.selected_indices: Optional[List[int]] = None
        self.selected_param_names: Optional[List[str]] = None
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None,
                source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        if self.strategy == 'none':
            logger.debug("Dimension selection disabled, returning input space")
            return input_space
        
        # Step 1: Get exclude_params indices
        all_param_names = input_space.get_hyperparameter_names()
        exclude_indices_set = set()
        for exclude_name in self.exclude_params:
            if exclude_name in all_param_names:
                exclude_indices_set.add(all_param_names.index(exclude_name))
        
        # Step 2: Get expert parameter indices
        expert_indices = self._get_expert_param_indices(input_space)
        expert_indices = [idx for idx in expert_indices if idx not in exclude_indices_set]
        
        # Step 3: Get method-selected parameters (sorted by importance)
        method_sorted_indices = self._select_parameters(input_space, space_history, source_similarities)
        # Filter out exclude_params and expert_params from method selection
        method_sorted_indices = [idx for idx in method_sorted_indices 
                                if idx not in exclude_indices_set and idx not in expert_indices]
        
        # Step 4: Merge: expert params as base, then supplement from method-sorted list
        selected_indices = self._merge_expert_and_method_params(
            expert_indices, method_sorted_indices, input_space
        )
        
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
    
    def _get_expert_param_indices(self, input_space: ConfigurationSpace) -> List[int]:
        if not self.expert_params:
            return []
        
        all_param_names = input_space.get_hyperparameter_names()
        expert_indices = []
        
        for param_name in self.expert_params:
            if param_name in all_param_names:
                idx = all_param_names.index(param_name)
                if idx not in expert_indices:
                    expert_indices.append(idx)
                    logger.debug(f"Including expert parameter: {param_name}")
            else:
                logger.warning(f"Expert parameter '{param_name}' not found in configuration space")
        
        return expert_indices
    
    def _get_target_topk(self) -> Optional[int]:
        if hasattr(self, 'topk') and self.topk > 0:
            return self.topk
        elif hasattr(self, 'current_topk') and self.current_topk > 0:
            return self.current_topk
        return None
    
    def _merge_expert_and_method_params(self,
                                       expert_indices: List[int],
                                       method_sorted_indices: List[int],
                                       input_space: ConfigurationSpace) -> List[int]:
        """
        Merge expert parameters and method-selected parameters.
        
        Strategy:
        1. Expert parameters as base (already filtered by exclude_params)
        2. Supplement from method_sorted_indices (sorted by importance) until reaching topk
        
        Args:
            expert_indices: Expert parameter indices (already filtered)
            method_sorted_indices: Method-selected parameter indices sorted by importance (already filtered)
            input_space: Input configuration space
            
        Returns:
            Merged list of parameter indices
        """
        merged_indices = expert_indices.copy()
        
        target_topk = self._get_target_topk()
        
        if target_topk is not None:
            for idx in method_sorted_indices:
                if len(merged_indices) >= target_topk:
                    break
                if idx not in merged_indices:
                    merged_indices.append(idx)
        else:
            for idx in method_sorted_indices:
                if idx not in merged_indices:
                    merged_indices.append(idx)
        
        if expert_indices:
            n_expert = len(expert_indices)
            n_method_added = len(merged_indices) - n_expert
            n_merged = len(merged_indices)
            target_str = f" (target: {target_topk})" if target_topk else ""
            logger.info(f"{self.strategy} dimension selection: {n_expert} expert + {n_method_added} method = {n_merged} total{target_str}")
        
        return sorted(merged_indices)
    
    def _select_parameters(self, 
                          input_space: ConfigurationSpace,
                          space_history: Optional[List[History]] = None,
                          source_similarities: Optional[Dict[int, float]] = None) -> List[int]:
        """
        Select parameters to keep using the specific method (e.g., SHAP, correlation).
        
        Subclasses should override this method.
        Note: Expert parameters will be automatically merged by the base class.
        
        Args:
            input_space: Input configuration space
            space_history: Historical data
            source_similarities: Source task similarities
            
        Returns:
            List of selected parameter indices (excluding expert params, which are handled separately)
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
    
    def _apply_exclude_params(self, 
                              selected_indices: List[int],
                              input_space: ConfigurationSpace,
                              step_name: str = "dimension selection") -> List[int]:
        if not self.exclude_params:
            return selected_indices
        
        all_param_names = input_space.get_hyperparameter_names()
        result_indices = selected_indices.copy()
        excluded_count = 0
        
        for exclude_name in self.exclude_params:
            if exclude_name in all_param_names:
                exclude_idx = all_param_names.index(exclude_name)
                if exclude_idx in result_indices:
                    result_indices.remove(exclude_idx)
                    excluded_count += 1
                    logger.debug(f"Excluded parameter '{exclude_name}' from {step_name}")
                else:
                    logger.debug(f"Parameter '{exclude_name}' was not in selected parameters, skipping exclusion")
            else:
                logger.warning(f"Exclude parameter '{exclude_name}' not found in configuration space")
        
        if excluded_count > 0:
            logger.info(f"{step_name}: Excluded {excluded_count} parameter(s) from selection")
        return sorted(result_indices)
    
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
        if self.expert_params:
            info['expert_params'] = self.expert_params
            info['n_expert_params'] = len(self.expert_params)
        if self.exclude_params:
            info['exclude_params'] = self.exclude_params
        if self.input_space:
            info['compression_ratio'] = len(self.selected_indices) / len(self.input_space.get_hyperparameters())
        return info