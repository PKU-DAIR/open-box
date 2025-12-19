import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from ...utils import (
    extract_numeric_hyperparameters,
    extract_top_samples_from_history,
)


class ImportanceCalculator(ABC):    
    @abstractmethod
    def calculate_importances(self,
                             input_space: ConfigurationSpace,
                             space_history: Optional[List[History]] = None,
                             source_similarities: Optional[List[Tuple[int, float]]] = None) -> Tuple[List[str], np.ndarray]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class SHAPImportanceCalculator(ImportanceCalculator):    
    def __init__(self):
        self._cache = {
            'models': None,
            'importances': None,
            'shap_values': None,
            'n_features': None,
        }
        self.numeric_hyperparameter_names: List[str] = []
        self.numeric_hyperparameter_indices: List[int] = []
    
    def calculate_importances(self,
                             input_space: ConfigurationSpace,
                             space_history: Optional[List[History]] = None,
                             source_similarities: Optional[List[Tuple[int, float]]] = None) -> Tuple[List[str], np.ndarray]:
        self.numeric_hyperparameter_names, \
        self.numeric_hyperparameter_indices = extract_numeric_hyperparameters(input_space)
        
        if not space_history:
            logger.warning("No space history provided for SHAP, returning uniform importances")
            return self.numeric_hyperparameter_names, np.ones(len(self.numeric_hyperparameter_names))
        
        current_n_features = len(self.numeric_hyperparameter_names)
        cache_valid = (
            self._cache['models'] is not None and
            self._cache['importances'] is not None and
            self._cache['n_features'] == current_n_features
        )
        
        if cache_valid:
            logger.info(f"Using cached SHAP model (n_features={current_n_features})")
            return self.numeric_hyperparameter_names, self._cache['importances']
        
        importances = self._compute_shap_importances(space_history, input_space, source_similarities)
        
        if importances is None or np.size(importances) == 0:
            logger.warning("SHAP importances unavailable, returning uniform importances")
            return self.numeric_hyperparameter_names, np.ones(len(self.numeric_hyperparameter_names))
        
        return self.numeric_hyperparameter_names, importances
    
    def _compute_shap_importances(self,
                                 space_history: List[History],
                                 input_space: ConfigurationSpace,
                                 source_similarities: Optional[List[Tuple[int, float]]] = None) -> np.ndarray:
        import shap
        from sklearn.ensemble import RandomForestRegressor
        
        models = []
        importances_list = []
        shap_values = []
        
        if len(space_history) == 0:
            logger.warning("No historical data provided for SHAP")
            return None
        
        all_x, all_y = extract_top_samples_from_history(
            space_history, self.numeric_hyperparameter_names, input_space,
            top_ratio=1.0, normalize=True
        )
        
        for task_idx, (hist_x_numeric, hist_y) in enumerate(zip(all_x, all_y)):
            if len(hist_x_numeric) == 0:
                continue
            
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
            logger.debug(f"SHAP importance (task {task_idx}): {df.to_string()}")
        
        if len(importances_list) == 0:
            logger.warning("No SHAP importances computed")
            return None
        
        importances_array = np.array(importances_list)
        if source_similarities:
            similarity_dict = {idx: sim for idx, sim in source_similarities}
            weights = np.array([
                similarity_dict.get(task_idx, 0.0) 
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
        
        self._cache.update({
            'models': models,
            'importances': importances,
            'shap_values': shap_values,
            'n_features': len(self.numeric_hyperparameter_names),
        })
        
        return importances
    
    def get_name(self) -> str:
        return "SHAP"


class CorrelationImportanceCalculator(ImportanceCalculator):    
    def __init__(self, method: str = 'spearman'):
        """
        Args:
            method: 'spearman' or 'pearson'
        """
        self.method = method
    
    def calculate_importances(self,
                             input_space: ConfigurationSpace,
                             space_history: Optional[List[History]] = None,
                             source_similarities: Optional[List[Tuple[int, float]]] = None) -> Tuple[List[str], np.ndarray]:
        numeric_param_names, _ = extract_numeric_hyperparameters(input_space)
        
        if not space_history:
            logger.warning("No space history provided for correlation, returning uniform importances")
            return numeric_param_names, np.ones(len(numeric_param_names))
        
        all_x, all_y = extract_top_samples_from_history(
            space_history, numeric_param_names, input_space,
            top_ratio=1.0, normalize=True
        )
        
        if len(all_x) == 0:
            logger.warning("No data available for correlation")
            return numeric_param_names, np.ones(len(numeric_param_names))
        
        if len(all_x) > 1 and source_similarities:
            importances = self._compute_weighted_correlations(
                all_x, all_y, numeric_param_names, source_similarities
            )
        else:
            importances = self._compute_pooled_correlations(
                all_x, all_y, numeric_param_names
            )
        
        df = pd.DataFrame({
            "feature": numeric_param_names,
            "importance": importances,
        }).sort_values("importance", ascending=True)
        logger.debug(f"{self.method.capitalize()} correlation importance: {df.to_string()}")
        
        return numeric_param_names, importances
    
    def _compute_pooled_correlations(self, all_x, all_y, numeric_param_names) -> np.ndarray:
        from scipy.stats import spearmanr, pearsonr
        X = np.vstack(all_x)
        y = np.hstack(all_y)
        if len(X) < 2:
            logger.warning("Insufficient data for correlation")
            return np.ones(len(numeric_param_names))
        
        importances = []
        for i in range(X.shape[1]):
            if self.method == 'spearman':
                corr, _ = spearmanr(X[:, i], y)
            else:
                corr, _ = pearsonr(X[:, i], y)
            importances.append(-abs(corr) if not np.isnan(corr) else 0.0)
        return np.array(importances)
    
    def _compute_weighted_correlations(self, all_x, all_y, numeric_param_names,
                                    source_similarities) -> np.ndarray:
        from scipy.stats import spearmanr, pearsonr
        
        similarity_dict = {idx: sim for idx, sim in source_similarities}
        
        correlations_list = []
        for task_idx, (hist_x_numeric, hist_y) in enumerate(zip(all_x, all_y)):
            if len(hist_x_numeric) == 0:
                continue
            
            n_features = hist_x_numeric.shape[1]
            correlations = np.zeros(n_features)
            
            for i in range(n_features):
                try:
                    if self.method == 'spearman':
                        corr, _ = spearmanr(hist_x_numeric[:, i], hist_y.flatten())
                    else:
                        corr, _ = pearsonr(hist_x_numeric[:, i], hist_y.flatten())
                    
                    correlations[i] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception as e:
                    logger.warning(f"Failed to compute {self.method} correlation for feature {i}: {e}")
                    correlations[i] = 0.0
            
            correlations_list.append(correlations)
            
            df = pd.DataFrame({
                "feature": numeric_param_names,
                "correlation": correlations,
            }).sort_values("correlation", ascending=False)
            logger.debug(f"{self.method.capitalize()} correlations (task {task_idx}): {df.to_string()}")
        
        if len(correlations_list) == 0:
            logger.warning("No correlations computed")
            return np.ones(len(numeric_param_names))
        
        correlations_array = np.array(correlations_list)
        weights = np.array([
            similarity_dict.get(task_idx, 0.0) 
            for task_idx in range(len(correlations_list))
        ])
        weights_sum = weights.sum()
        
        if weights_sum > 1e-10:
            weights = weights / weights_sum
        else:
            weights = np.ones(len(correlations_list)) / len(correlations_list)
        
        correlations = np.average(correlations_array, axis=0, weights=weights)
        return -correlations
    
    def get_name(self) -> str:
        return f"Correlation({self.method})"