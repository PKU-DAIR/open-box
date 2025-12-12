import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from scipy.stats import spearmanr

from .base import DimensionSelectionStep
from ...utils import (
    extract_numeric_hyperparameters,
    extract_top_samples_from_history,
)


class SpearmanDimensionStep(DimensionSelectionStep):    
    def __init__(self, 
                 strategy: str = 'spearman', 
                 topk: int = 20,
                 source_similarities: Optional[List[Tuple[int, float]]] = None,
                 **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.topk = 0 if strategy == 'none' else topk

        self._correlation_cache = {
            'correlations': None,
        }
        
        self.numeric_hyperparameter_names: List[str] = []
        self.numeric_hyperparameter_indices: List[int] = []
        
        self.source_similarities = source_similarities or []
        if self.source_similarities:
            total_sim = sum(sim for _, sim in self.source_similarities)
            if total_sim > 0:
                self._similarity_dict = {idx: sim / total_sim for idx, sim in self.source_similarities}
            else:
                n_histories = len(self.source_similarities)
                self._similarity_dict = {idx: 1.0 / n_histories for idx, _ in self.source_similarities}
        else:
            self._similarity_dict = {}
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        self._extract_numeric_hyperparameters(input_space)
        return super().compress(input_space, space_history)
    
    def _extract_numeric_hyperparameters(self, input_space: ConfigurationSpace):
        self.numeric_hyperparameter_names, \
        self.numeric_hyperparameter_indices \
            = extract_numeric_hyperparameters(input_space)
        
    def _select_parameters(self, 
                        input_space: ConfigurationSpace,
                        space_history: Optional[List[History]] = None) -> List[int]:
        if self.topk <= 0:
            logger.warning("No topk provided for Spearman selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        if not space_history:
            logger.warning("No space history provided for Spearman selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        correlations = self._compute_spearman_correlations(space_history, input_space)
        
        if correlations is None or np.size(correlations) == 0:
            logger.warning("Spearman correlations unavailable, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        top_k = min(self.topk, len(self.numeric_hyperparameter_names))
        if top_k == 0:
            logger.warning("No numeric hyperparameters detected, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        selected_numeric_indices = np.argsort(-np.abs(correlations))[:top_k].tolist()
        selected_param_names = [self.numeric_hyperparameter_names[i] for i in selected_numeric_indices]
        correlations_selected = correlations[selected_numeric_indices]
        
        selected_indices = [self.numeric_hyperparameter_indices[i] for i in selected_numeric_indices]
        
        logger.debug(f"Spearman dimension selection: {selected_param_names}")
        logger.debug(f"Spearman correlations: {correlations_selected}")
        
        return selected_indices
    
    def _compute_spearman_correlations(self, 
                                      space_history: List[History],
                                      input_space: ConfigurationSpace):
        if (self._correlation_cache['correlations'] is not None):
            logger.info("Using cached Spearman correlations")
            return self._correlation_cache['correlations']

        if len(space_history) == 0:
            logger.warning("No historical data provided for Spearman")
            return None
        
        all_x, all_y = extract_top_samples_from_history(
            space_history, self.numeric_hyperparameter_names, input_space,
            top_ratio=1.0, normalize=True
        )
        
        correlations_list = []
        for task_idx, (hist_x_numeric, hist_y) in enumerate(zip(all_x, all_y)):
            if len(hist_x_numeric) == 0:
                continue
            
            n_features = hist_x_numeric.shape[1]
            correlations = np.zeros(n_features)
            
            for i in range(n_features):
                try:
                    corr, p_value = spearmanr(hist_x_numeric[:, i], hist_y.flatten())
                    if np.isnan(corr):
                        correlations[i] = 0.0
                    else:
                        correlations[i] = abs(corr)
                except Exception as e:
                    logger.warning(f"Failed to compute Spearman correlation for feature {i}: {e}")
                    correlations[i] = 0.0
            
            correlations_list.append(correlations)
            
            df = pd.DataFrame({
                "feature": self.numeric_hyperparameter_names,
                "correlation": correlations,
            }).sort_values("correlation", ascending=False)
            logger.debug(f"Spearman dimension compression correlations (task {task_idx}): {df.to_string()}")
        
        if len(correlations_list) == 0:
            logger.warning("No Spearman correlations computed")
            return None
        
        correlations_array = np.array(correlations_list)
        if self._similarity_dict:
            weights = np.array([
                self._similarity_dict.get(task_idx, 0.0) 
                for task_idx in range(len(correlations_list))
            ])
            weights_sum = weights.sum()
            if weights_sum > 1e-10:
                weights = weights / weights_sum
            else:
                weights = np.ones(len(correlations_list)) / len(correlations_list)
        else:
            weights = np.ones(len(correlations_list)) / len(correlations_list)
        
        correlations = np.average(correlations_array, axis=0, weights=weights)
        
        self._correlation_cache['correlations'] = correlations
        return correlations