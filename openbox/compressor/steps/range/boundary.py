import copy
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .base import RangeCompressionStep
from ...utils import (
    create_space_from_ranges,
    extract_numeric_hyperparameters,
    extract_top_samples_from_history,
)
from ...sampling import MixedRangeSamplingStrategy


class BoundaryRangeStep(RangeCompressionStep):
    @staticmethod
    def _clamp_range_bounds(min_val: float, max_val: float, 
                            param_values: np.ndarray,
                            original_space: ConfigurationSpace,
                            param_name: str) -> Tuple[float, float]:
        if min_val > max_val:
            min_val = np.min(param_values)
            max_val = np.max(param_values)
        
        hp = original_space.get_hyperparameter(param_name)
        original_min = hp.lower
        original_max = hp.upper
        
        min_val = max(min_val, original_min)
        max_val = min(max_val, original_max)
        return min_val, max_val
    
    def __init__(self, 
                 method: str = 'boundary',
                 top_ratio: float = 0.8,
                 sigma: float = 2.0,
                 enable_mixed_sampling: bool = True,
                 initial_prob: float = 0.9,
                 use_shap: bool = True,
                 seed: Optional[int] = None,
                 **kwargs):
        super().__init__(method=method, **kwargs)
        self.top_ratio = top_ratio
        self.sigma = sigma
        self.enable_mixed_sampling = enable_mixed_sampling
        self.initial_prob = initial_prob
        self.use_shap = use_shap
        self.seed = seed

    
    def _compute_compressed_space(self, 
                                input_space: ConfigurationSpace,
                                space_history: Optional[List[History]] = None,
                                source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        if not space_history:
            logger.warning("No space history provided for boundary compression, returning input space")
            return copy.deepcopy(input_space)
        
        numeric_param_names, numeric_param_indices = extract_numeric_hyperparameters(input_space)
        
        if not numeric_param_names:
            logger.warning("No numeric hyperparameters found, returning input space")
            return copy.deepcopy(input_space)
        
        compressed_ranges = self._compute_simple_ranges(
            space_history, numeric_param_names, input_space
        )

        compressed_space = create_space_from_ranges(input_space, compressed_ranges)
        logger.info(f"Boundary range compression: {len(compressed_ranges)} parameters compressed")
        return compressed_space
        
    def _compute_simple_ranges(self, 
                            space_history: List[History],
                            numeric_param_names: List[str],
                            original_space: ConfigurationSpace) -> Dict[str, Tuple[float, float]]:
        all_x, _ = extract_top_samples_from_history(
            space_history, numeric_param_names, original_space,
            top_ratio=self.top_ratio, normalize=True
        )
        
        if len(all_x) == 0:
            return {}
        
        X_combined = np.vstack(all_x)
        
        compressed_ranges = {}
        for i, param_name in enumerate(numeric_param_names):
            values_norm = X_combined[:, i]
            
            mean = np.mean(values_norm)
            std = np.std(values_norm)
            min_val_norm = max(np.min(values_norm), mean - self.sigma * std)
            max_val_norm = min(np.max(values_norm), mean + self.sigma * std)
            
            hp = original_space.get_hyperparameter(param_name)
            lower = hp.lower
            upper = hp.upper
            range_size = upper - lower
            
            min_val = lower + min_val_norm * range_size
            max_val = lower + max_val_norm * range_size
            
            values_original = lower + values_norm * range_size
            
            min_val, max_val = self._clamp_range_bounds(
                min_val, max_val, values_original, original_space, param_name
            )
            
            compressed_ranges[param_name] = (min_val, max_val)
        
        return compressed_ranges
    
    def get_sampling_strategy(self):
        if self.enable_mixed_sampling and self.original_space is not None:
            compressed_space = self.output_space if self.output_space else self.original_space
            return MixedRangeSamplingStrategy(
                compressed_space=compressed_space,
                original_space=self.original_space,
                initial_prob=self.initial_prob,
                method='boundary',
                seed=self.seed
            )
        return None

