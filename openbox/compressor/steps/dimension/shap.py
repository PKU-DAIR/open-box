import numpy as np
import pandas as pd
import shap
from typing import Optional, List
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .base import DimensionSelectionStep
from ...utils import prepare_historical_data


class SHAPDimensionStep(DimensionSelectionStep):    
    def __init__(self, strategy: str = 'shap', topk: int = 20, **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.topk = 0 if strategy == 'none' else topk

        self._shap_cache = {
            'models': None,
            'importances': None,
            'shap_values': None,
        }
        
        self.numeric_hyperparameter_names: List[str] = []
        self.numeric_hyperparameter_indices: List[int] = []
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None) -> ConfigurationSpace:
        self._extract_numeric_hyperparameters(input_space)
        return super().compress(input_space, space_history)
    
    def _extract_numeric_hyperparameters(self, input_space: ConfigurationSpace):
        param_names = input_space.get_hyperparameter_names()
        self.numeric_hyperparameter_names = []
        self.numeric_hyperparameter_indices = []
        
        for i, name in enumerate(param_names):
            hp = input_space.get_hyperparameter(name)
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                self.numeric_hyperparameter_names.append(name)
                self.numeric_hyperparameter_indices.append(i)
    
    def _select_parameters(self, 
                        input_space: ConfigurationSpace,
                        space_history: Optional[List[History]] = None) -> List[int]:
        if self.topk <= 0:
            logger.warning("No topk provided for SHAP selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        if not space_history:
            logger.warning("No space history provided for SHAP selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        hist_x, hist_y = prepare_historical_data(space_history)
        
        if len(hist_x) == 0 or len(hist_y) == 0:
            logger.warning("No valid historical data for SHAP selection, keeping all parameters")
            return list(range(len(input_space.get_hyperparameters())))
        
        _, importances, _ = self._compute_shap_importances(hist_x, hist_y)
        
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
        
        # Convert numeric parameter indices to global parameter indices
        selected_indices = [self.numeric_hyperparameter_indices[i] for i in selected_numeric_indices]
        
        logger.debug(f"SHAP dimension selection: {selected_param_names}")
        logger.debug(f"SHAP importances: {importances_selected}")
        
        return selected_indices
    
    def _compute_shap_importances(self, 
                                hist_x: List[np.ndarray], 
                                hist_y: List[np.ndarray]):
        if (self._shap_cache['models'] is not None and
            self._shap_cache['importances'] is not None):
            logger.info("Using cached SHAP model")
            return (self._shap_cache['models'],
                    self._shap_cache['importances'],
                    self._shap_cache['shap_values'])
        
        models = []
        importances = []
        shap_values = []
        
        if len(hist_x) == 0:
            logger.warning("No historical data provided for SHAP")
            return models, None, shap_values
        
        for i in range(len(hist_x)):
            if hist_x[i].shape[1] == len(self.numeric_hyperparameter_names):
                hist_x_numeric = hist_x[i]
            else:
                hist_x_numeric = hist_x[i][:, self.numeric_hyperparameter_indices]
                hist_x_numeric = hist_x_numeric.astype(float)
            
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(hist_x_numeric, hist_y[i])
            
            explainer = shap.Explainer(model)
            shap_value = -np.abs(explainer(hist_x_numeric, check_additivity=False).values)
            mean_shap = shap_value.mean(axis=0)
            
            models.append(model)
            importances.append(mean_shap)
            shap_values.append(shap_value)
            df = pd.DataFrame({
                "feature": self.numeric_hyperparameter_names,
                "importance": mean_shap,
                "effect": np.where(mean_shap < 0, "increase_objective", "decrease_objective")
            }).sort_values("importance", ascending=True)
            logger.debug(f"SHAP dimension compression feature importance: {df.to_string()}")
        
        # Average importances across tasks
        if len(importances) == 0:
            logger.warning("No SHAP importances computed")
            return models, None, shap_values
        
        importances = np.mean(np.array(importances), axis=0)
        
        self._shap_cache.update({
            'models': models,
            'importances': importances,
            'shap_values': shap_values,
        })
        
        return models, importances, shap_values
