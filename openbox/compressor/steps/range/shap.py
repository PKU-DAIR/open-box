import copy
import numpy as np
from typing import Optional, List, Tuple, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from sklearn.ensemble import RandomForestRegressor
import shap

from .boundary import BoundaryRangeStep
from ...utils import (
    create_space_from_ranges,
    extract_numeric_hyperparameters,
    extract_top_samples_from_history,
)


class SHAPBoundaryRangeStep(BoundaryRangeStep):    
    def __init__(self,
                 method: str = 'shap_boundary',
                 top_ratio: float = 0.8,
                 sigma: float = 2.0,
                 enable_mixed_sampling: bool = True,
                 initial_prob: float = 0.9,
                 seed: Optional[int] = None,
                 **kwargs):
        super().__init__(
            method=method,
            top_ratio=top_ratio,
            sigma=sigma,
            enable_mixed_sampling=enable_mixed_sampling,
            initial_prob=initial_prob,
            use_shap=True,
            seed=seed,
            **kwargs
        )
    
    def _compute_compressed_space(self,
                                  input_space: ConfigurationSpace,
                                  space_history: Optional[List[History]] = None,
                                  source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        if not space_history:
            logger.warning("No space history provided for SHAP boundary compression, returning input space")
            return copy.deepcopy(input_space)
        
        numeric_param_names, numeric_param_indices = extract_numeric_hyperparameters(input_space)
        
        if not numeric_param_names:
            logger.warning("No numeric hyperparameters found, returning input space")
            return copy.deepcopy(input_space)
        
        compressed_ranges = self._compute_shap_based_ranges(
            space_history, numeric_param_names, input_space, source_similarities
        )
        
        compressed_space = create_space_from_ranges(input_space, compressed_ranges)
        logger.info(f"SHAP boundary range compression: {len(compressed_ranges)} parameters compressed")
        
        return compressed_space
    
    
    def _compute_shap_based_ranges(self,
                                   space_history: List[History],
                                   numeric_param_names: List[str],
                                   original_space: ConfigurationSpace,
                                   source_similarities: Optional[Dict[int, float]] = None) -> Dict[str, Tuple[float, float]]:        
        all_x, all_y, sample_history_indices = extract_top_samples_from_history(
            space_history, numeric_param_names, original_space,
            top_ratio=self.top_ratio, normalize=True, return_history_indices=True
        )
        
        if len(all_x) == 0:
            return {}
        
        X_combined = np.vstack(all_x)
        y_combined = np.concatenate(all_y)
        sample_history_indices = np.array(sample_history_indices)
        
        model = RandomForestRegressor(n_estimators=100, random_state=self.seed or 42)
        model.fit(X_combined, y_combined)

        explainer = shap.Explainer(model)
        shap_values = explainer(X_combined)
        shap_vals_array = -np.abs(shap_values.values)
        logger.debug(f"SHAP values: {shap_vals_array}")
        
        compressed_ranges = {}
        
        for i, param_name in enumerate(numeric_param_names):
            param_shap = shap_vals_array[:, i]  # Original SHAP values (can be negative)
            param_values = X_combined[:, i]

            beneficial_mask = param_shap < 0
            beneficial_values = param_values[beneficial_mask]
            beneficial_shap = param_shap[beneficial_mask]
            beneficial_history_indices = sample_history_indices[beneficial_mask]
            
            if len(beneficial_values) == 0:
                logger.warning(
                    f"Parameter {param_name} has no samples with SHAP < 0. "
                    f"Using all samples with uniform weights."
                )
                beneficial_values = param_values
                beneficial_shap = np.ones_like(param_values)
                beneficial_history_indices = sample_history_indices
            else:
                logger.debug(
                    f"Parameter {param_name}: {len(beneficial_values)}/{len(param_values)} samples "
                    f"have SHAP < 0 (beneficial)"
                )
                beneficial_shap = -beneficial_shap  # Convert negative to positive weights
            
            if source_similarities:
                beneficial_similarities = np.array([
                    source_similarities.get(idx, 0.0) for idx in beneficial_history_indices
                ])
            
            # Combined weight = SHAP weight * similarity weight
            combined_weights = beneficial_shap * beneficial_similarities
            
            combined_weights_sum = combined_weights.sum()
            if combined_weights_sum < 1e-10:
                unique_values = len(np.unique(beneficial_values))
                logger.warning(
                    f"Parameter {param_name} has zero combined weights. "
                    f"Using uniform weights for {len(beneficial_values)} beneficial samples."
                )
                weights = np.ones_like(combined_weights) / len(combined_weights)
            else:
                weights = combined_weights / combined_weights_sum
            
            weighted_mean = np.average(beneficial_values, weights=weights)
            weighted_std = np.sqrt(np.average((beneficial_values - weighted_mean) ** 2, weights=weights))
            
            min_val_norm = max(np.min(beneficial_values), weighted_mean - self.sigma * weighted_std)
            max_val_norm = min(np.max(beneficial_values), weighted_mean + self.sigma * weighted_std)
            
            hp = original_space.get_hyperparameter(param_name)
            lower = hp.lower
            upper = hp.upper
            range_size = upper - lower
            
            min_val = lower + min_val_norm * range_size
            max_val = lower + max_val_norm * range_size
            
            beneficial_values_original = lower + beneficial_values * range_size
            
            min_val, max_val = self._clamp_range_bounds(
                min_val, max_val, beneficial_values_original, original_space, param_name
            )
            compressed_ranges[param_name] = (min_val, max_val)
        
        logger.info(f"SHAP-based ranges computed for {len(compressed_ranges)} parameters")
        return compressed_ranges

