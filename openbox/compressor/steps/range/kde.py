import copy
import numpy as np
from typing import Optional, List, Tuple, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from scipy.stats import gaussian_kde

from .boundary import BoundaryRangeStep
from ...utils import (
    create_space_from_ranges,
    extract_numeric_hyperparameters,
    extract_numeric_values_from_configs,
)


class KDEBoundaryRangeStep(BoundaryRangeStep):    
    def __init__(self,
                 method: str = 'kde_boundary',
                 source_top_ratio: float = 0.3,  # Top 30% of source task configs
                 kde_coverage: float = 0.6,  # Coverage ratio for KDE interval
                 enable_mixed_sampling: bool = True,
                 initial_prob: float = 0.9,
                 seed: Optional[int] = None,
                 top_ratio: Optional[float] = None,
                 **kwargs):
        if top_ratio is None:
            top_ratio = source_top_ratio
        
        super().__init__(
            method=method,
            top_ratio=top_ratio,
            sigma=2.0,  # Not used in KDE method
            enable_mixed_sampling=enable_mixed_sampling,
            initial_prob=initial_prob,
            use_shap=False,
            seed=seed,
            **kwargs
        )
        self.source_top_ratio = source_top_ratio
        self.kde_coverage = kde_coverage

    
    def _compute_compressed_space(self,
                                input_space: ConfigurationSpace,
                                space_history: Optional[List[History]] = None,
                                source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        if not space_history:
            logger.warning("No space history provided for KDE boundary compression, returning input space")
            return copy.deepcopy(input_space)

        numeric_param_names, _ = extract_numeric_hyperparameters(input_space)
        
        if not numeric_param_names:
            logger.warning("No numeric hyperparameters found, returning input space")
            return copy.deepcopy(input_space)
        
        compressed_ranges = self._compute_kde_based_ranges(
            space_history, numeric_param_names, input_space, source_similarities
        )
        
        compressed_space = create_space_from_ranges(input_space, compressed_ranges)
        logger.info(f"KDE boundary range compression: {len(compressed_ranges)} parameters compressed")
        
        return compressed_space
    
    def _compute_kde_based_ranges(self,
                                space_history: List[History],
                                numeric_param_names: List[str],
                                original_space: ConfigurationSpace,
                                source_similarities: Optional[Dict[int, float]] = None) -> Dict[str, Tuple[float, float]]:
        median_top_ratio = self.source_top_ratio
        
        compressed_ranges = {}
        
        for param_name in numeric_param_names:
            weighted_values = []
            weights = []
            
            for task_idx, history in enumerate(space_history):
                if len(history) == 0:
                    continue
                
                if source_similarities:
                    similarity = source_similarities.get(task_idx, 0.0)
                    if similarity <= 0:
                        continue
                else:
                    n_histories = len(space_history)
                    similarity = 1.0 / n_histories if n_histories > 0 else 0.0
                    if similarity <= 0:
                        continue
                
                objectives = history.get_objectives()
                if len(objectives) == 0:
                    continue
                
                sorted_indices = np.argsort(objectives.flatten())
                top_n = max(1, int(len(sorted_indices) * median_top_ratio))
                top_indices = sorted_indices[: top_n]
                
                top_configs = [history.observations[idx].config for idx in top_indices]
                param_values = extract_numeric_values_from_configs(
                    top_configs, [param_name], original_space, normalize=False
                )
                for rank, value in enumerate(param_values[:, 0]):
                    if np.isnan(value):
                        continue
                    
                    weight = (rank + 1) * similarity
                    weighted_values.append(float(value))
                    weights.append(weight)
            
            if len(weighted_values) == 0:
                logger.warning(f"No weighted values for parameter {param_name}, skipping")
                continue
            
            weighted_values = np.array(weighted_values)
            weights = np.array(weights)
            
            weights = weights / (weights.sum() + 1e-10)
            
            try:
                kde = gaussian_kde(weighted_values, weights=weights)
            except Exception as e:
                logger.warning(f"Failed to build KDE for {param_name}: {e}, using simple range")
                min_val = np.min(weighted_values)
                max_val = np.max(weighted_values)
                hp = original_space.get_hyperparameter(param_name)
                compressed_ranges[param_name] = (
                    max(min_val, hp.lower),
                    min(max_val, hp.upper)
                )
                continue
            
            hp = original_space.get_hyperparameter(param_name)
            original_min = hp.lower
            original_max = hp.upper
            
            grid_size = 1000
            grid = np.linspace(original_min, original_max, grid_size)
            kde_density = kde(grid)
            
            kde_density = kde_density / (kde_density.sum() + 1e-10)
            
            sorted_indices = np.argsort(kde_density)[::-1]
            cumulative_density = np.cumsum(kde_density[sorted_indices])
            
            n_points_needed = np.searchsorted(cumulative_density, self.kde_coverage) + 1
            n_points_needed = min(n_points_needed, len(grid))
            
            selected_indices = sorted_indices[:n_points_needed]
            selected_grid = grid[selected_indices]
            
            min_val = np.min(selected_grid)
            max_val = np.max(selected_grid)
            
            min_val, max_val = self._clamp_range_bounds(
                min_val, max_val, weighted_values, original_space, param_name
            )
            
            compressed_ranges[param_name] = (min_val, max_val)
        
        logger.info(f"KDE-based ranges computed for {len(compressed_ranges)} parameters")
        return compressed_ranges

