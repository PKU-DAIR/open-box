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
    prepare_historical_data,
    create_space_from_ranges,
    extract_numeric_hyperparameters,
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
                                  space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        if not space_history:
            logger.warning("No space history provided for SHAP boundary compression, returning input space")
            return copy.deepcopy(input_space)
        
        hist_x, hist_y = prepare_historical_data(space_history)
        
        if len(hist_x) == 0 or len(hist_y) == 0:
            logger.warning("No valid historical data for SHAP boundary compression, returning input space")
            return copy.deepcopy(input_space)
        
        numeric_param_names, numeric_param_indices = extract_numeric_hyperparameters(input_space)
        
        if not numeric_param_names:
            logger.warning("No numeric hyperparameters found, returning input space")
            return copy.deepcopy(input_space)
        
        compressed_ranges = self._compute_shap_based_ranges(
            hist_x, hist_y, numeric_param_names, numeric_param_indices, input_space
        )
        
        compressed_space = create_space_from_ranges(input_space, compressed_ranges)
        logger.info(f"SHAP boundary range compression: {len(compressed_ranges)} parameters compressed")
        
        return compressed_space
    
    def _compute_shap_based_ranges(self,
                                   hist_x: List[np.ndarray],
                                   hist_y: List[np.ndarray],
                                   numeric_param_names: List[str],
                                   numeric_param_indices: List[int],
                                   original_space: ConfigurationSpace) -> Dict[str, Tuple[float, float]]:        
        all_x = []
        all_y = []
        
        for i in range(len(hist_x)):
            if hist_x[i].shape[1] == len(numeric_param_names):
                x_numeric = hist_x[i]
            else:
                x_numeric = hist_x[i][:, numeric_param_indices].astype(float)
            
            sorted_indices = np.argsort(hist_y[i])
            top_n = int(len(sorted_indices) * self.top_ratio)
            top_indices = sorted_indices[: top_n]
            
            all_x.append(x_numeric[top_indices])
            all_y.append(hist_y[i][top_indices])
        
        if len(all_x) == 0:
            return {}
        
        X_combined = np.vstack(all_x)
        y_combined = np.concatenate(all_y)
        
        model = RandomForestRegressor(n_estimators=100, random_state=self.seed or 42)
        model.fit(X_combined, y_combined)

        explainer = shap.Explainer(model)
        shap_values = explainer(X_combined)
        shap_vals_array = -np.abs(shap_values.values)
        
        compressed_ranges = {}
        
        for i, param_name in enumerate(numeric_param_names):
            param_shap = shap_vals_array[:, i]
            
            param_values = X_combined[:, i]
            
            weights = param_shap / (param_shap.sum() + 1e-10)
            
            weighted_mean = np.average(param_values, weights=weights)
            weighted_std = np.sqrt(np.average((param_values - weighted_mean) ** 2, weights=weights))
            
            min_val = max(np.min(param_values), weighted_mean - self.sigma * weighted_std)
            max_val = min(np.max(param_values), weighted_mean + self.sigma * weighted_std)
            
            min_val, max_val = self._clamp_range_bounds(
                min_val, max_val, param_values, original_space, param_name
            )
            compressed_ranges[param_name] = (min_val, max_val)
        
        logger.info(f"SHAP-based ranges computed for {len(compressed_ranges)} parameters")
        return compressed_ranges
