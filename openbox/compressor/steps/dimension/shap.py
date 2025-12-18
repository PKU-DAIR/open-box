import numpy as np
import pandas as pd
import shap
from typing import Optional, List, Tuple
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .base import DimensionSelectionStep
from ...utils import (
    extract_numeric_hyperparameters,
    extract_top_samples_from_history,
)


class SHAPDimensionStep(DimensionSelectionStep):    
    def __init__(self, 
                 strategy: str = 'shap', 
                 topk: int = 20,
                 source_similarities: Optional[List[Tuple[int, float]]] = None,
                 **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.topk = 0 if strategy == 'none' else topk

        self._shap_cache = {
            'models': None,
            'importances': None,
            'shap_values': None,
            'n_features': None,
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
            logger.warning("No topk provided for SHAP selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        if not space_history:
            logger.warning("No space history provided for SHAP selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        _, importances, _ = self._compute_shap_importances(space_history, input_space)
        
        if importances is None or np.size(importances) == 0:
            logger.warning("SHAP importances unavailable, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        top_k = min(self.topk, len(self.numeric_hyperparameter_names))
        if top_k == 0:
            logger.warning("No numeric hyperparameters detected, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        selected_numeric_indices = np.argsort(importances)[: top_k].tolist()
        selected_param_names = [self.numeric_hyperparameter_names[i] for i in selected_numeric_indices]
        importances_selected = importances[selected_numeric_indices]
        
        selected_indices = [self.numeric_hyperparameter_indices[i] for i in selected_numeric_indices]
        
        logger.debug(f"SHAP dimension selection: {selected_param_names}")
        logger.debug(f"SHAP importances: {importances_selected}")
        
        return selected_indices
    
    def _compute_shap_importances(self, 
                                space_history: List[History],
                                input_space: ConfigurationSpace):
        current_n_features = len(self.numeric_hyperparameter_names)
        cache_valid = (
            self._shap_cache['models'] is not None and
            self._shap_cache['importances'] is not None and
            self._shap_cache['n_features'] == current_n_features
        )
        
        if cache_valid:
            logger.info(f"Using cached SHAP model (n_features={current_n_features})")
            return (self._shap_cache['models'],
                    self._shap_cache['importances'],
                    self._shap_cache['shap_values'])

        models = []
        importances = []
        shap_values = []
        
        if len(space_history) == 0:
            logger.warning("No historical data provided for SHAP")
            return models, None, shap_values
        
        all_x, all_y = extract_top_samples_from_history(
            space_history, self.numeric_hyperparameter_names, input_space,
            top_ratio=1.0, normalize=True
        )
        
        importances_list = []
        for task_idx, (hist_x_numeric, hist_y) in enumerate(zip(all_x, all_y)):
            if len(hist_x_numeric) == 0:
                continue
            
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(hist_x_numeric, hist_y)
            
            explainer = shap.Explainer(model)
            shap_value = -np.abs(explainer(hist_x_numeric, check_additivity=False).values)
            mean_shap = shap_value.mean(axis=0)
            
            models.append(model)
            importances_list.append(mean_shap)
            shap_values.append(shap_value)
            df = pd.DataFrame({
                "feature": self.numeric_hyperparameter_names,
                "importance": mean_shap,
            }).sort_values("importance", ascending=True)
            logger.debug(f"SHAP dimension compression feature importance (task {task_idx}): {df.to_string()}")
        
        if len(importances_list) == 0:
            logger.warning("No SHAP importances computed")
            return models, None, shap_values
        
        importances_array = np.array(importances_list)
        if self._similarity_dict:
            weights = np.array([
                self._similarity_dict.get(task_idx, 0.0) 
                for task_idx in range(len(importances_list))
            ])
            weights_sum = weights.sum()
            if weights_sum > 1e-10:
                weights = weights / weights_sum
            else:
                weights = np.ones(len(importances_list)) / len(importances_list)
        else:
            weights = np.ones(len(importances_list)) / len(importances_list)
        
        importances = np.average(importances_array, axis=0, weights=weights)
        
        self._shap_cache.update({
            'models': models,
            'importances': importances,
            'shap_values': shap_values,
            'n_features': len(self.numeric_hyperparameter_names),
        })
        
        return models, importances, shap_values