import copy
import numpy as np
from typing import Optional, List, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace, Configuration
import ConfigSpace as CS
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

from .base import TransformativeProjectionStep
from ...utils import (
    extract_numeric_hyperparameters,
    extract_top_samples_from_history,
)


class KPCAProjectionStep(TransformativeProjectionStep):    
    def __init__(self,
                 method: str = 'kpca',
                 n_components: int = 10,
                 kernel: str = 'rbf',
                 gamma: Optional[float] = None,
                 space_history: Optional[List[History]] = None,
                 seed: int = 42,
                 **kwargs):
        super().__init__(method=method, **kwargs)
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.seed = seed
        self._rs = np.random.RandomState(seed=seed)
        
        self._kpca: Optional[KernelPCA] = None
        self._scaler: Optional[StandardScaler] = None
        self.active_hps: List = []
        self.numeric_param_names: List[str] = []
        self.numeric_param_indices: List[int] = []
        self._projected_samples: Optional[np.ndarray] = None
        
        self.space_history = space_history
    
    def compress(self, input_space: ConfigurationSpace,
                space_history: Optional[List[History]] = None,
                source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        if self.method == 'none':
            logger.info("KPCA projection disabled, returning input space")
            return input_space
        
        if space_history is not None:
            self.space_history = space_history
        
        if not self.space_history:
            logger.warning("No space history provided for KPCA, returning input space")
            return input_space
        
        self.numeric_param_names, self.numeric_param_indices = extract_numeric_hyperparameters(input_space)
        self.active_hps = list(input_space.get_hyperparameters())
        
        if len(self.numeric_param_names) == 0:
            logger.warning("No numeric hyperparameters found for KPCA, returning input space")
            return input_space
        
        self._train_kpca(input_space)
        
        if self._kpca is None:
            logger.warning("Failed to train KPCA, returning input space")
            return input_space
        
        projected_space = self._build_projected_space(input_space)
        logger.info(f"KPCA projection: {len(self.numeric_param_names)} -> "
                f"{self.n_components} components (kernel={self.kernel})")
        return projected_space
    
    def _train_kpca(self, input_space: ConfigurationSpace):
        all_x, all_y = extract_top_samples_from_history(
            self.space_history, self.numeric_param_names, input_space,
            top_ratio=1.0, normalize=True
        )
        
        if len(all_x) == 0:
            logger.warning("No historical data available for KPCA training")
            return
        
        X_combined = np.vstack(all_x)
        
        if X_combined.shape[0] < self.n_components:
            logger.warning(
                f"Insufficient samples for KPCA: {X_combined.shape[0]} < {self.n_components}. "
                f"Using {X_combined.shape[0]} components instead."
            )
            n_components = X_combined.shape[0]
        else:
            n_components = min(self.n_components, X_combined.shape[1])
        
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_combined)
        
        try:
            self._kpca = KernelPCA(
                n_components=n_components,
                kernel=self.kernel,
                gamma=self.gamma,
                random_state=self.seed,
                fit_inverse_transform=True
            )
            self._kpca.fit(X_scaled)
            
            if hasattr(self._kpca, 'n_components_'):
                self.n_components = self._kpca.n_components_
            elif hasattr(self._kpca, 'alphas_'):
                self.n_components = len(self._kpca.alphas_)
            else:
                self.n_components = n_components
                logger.warning(f"Could not determine actual n_components, using requested: {n_components}")
            
            self._projected_samples = self._kpca.transform(X_scaled)
            
            logger.info(f"KPCA trained successfully: {X_scaled.shape[1]} features -> {self.n_components} components "
                        f"(kernel={self.kernel}, samples={X_scaled.shape[0]})")
        except Exception as e:
            logger.error(f"Failed to train KPCA: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self._kpca = None
    
    def _build_projected_space(self, input_space: ConfigurationSpace) -> ConfigurationSpace:
        target = CS.ConfigurationSpace(
            name=f"{input_space.name}_kpca",
            seed=self.seed
        )
        
        hps = [
            CS.UniformFloatHyperparameter(
                name=f'kpca_{idx}',
                lower=-np.sqrt(len(self.numeric_param_names)),
                upper=np.sqrt(len(self.numeric_param_names))
            )
            for idx in range(self.n_components)
        ]
        
        target.add_hyperparameters(hps)
        self.output_space = target
        
        return target
    
    def project_point(self, point) -> dict:
        if self._kpca is None or self._scaler is None:
            logger.warning("KPCA not trained, returning empty projection")
            return {}
        
        if isinstance(point, Configuration):
            point_dict = point.get_dictionary()
        elif isinstance(point, dict):
            point_dict = point
        else:
            logger.warning(f"Unsupported point type: {type(point)}")
            return {}
        
        numeric_values = []
        for param_name in self.numeric_param_names:
            if param_name in point_dict:
                value = point_dict[param_name]
                hp = self.input_space.get_hyperparameter(param_name)
                if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                    normalized = (value - hp.lower) / (hp.upper - hp.lower)
                    numeric_values.append(normalized)
                else:
                    numeric_values.append(0.0)
            else:
                numeric_values.append(0.0)
        
        if len(numeric_values) != len(self.numeric_param_names):
            logger.warning(f"Mismatch in numeric values: {len(numeric_values)} != {len(self.numeric_param_names)}")
            return {}
        
        X = np.array([numeric_values])
        X_scaled = self._scaler.transform(X)
        X_kpca = self._kpca.transform(X_scaled)[0]
        return {f'kpca_{idx}': float(X_kpca[idx]) for idx in range(len(X_kpca))}
    
    def unproject_point(self, point: Configuration) -> dict:
        """
        KPCA does not need unprojection for evaluation.
        Sampling happens in original space.
        """
        if isinstance(point, Configuration):
            return point.get_dictionary()
        elif isinstance(point, dict):
            return point
        else:
            return dict(point)
    
    def needs_unproject(self) -> bool:
        return False
    
    def affects_sampling_space(self) -> bool:
        # does not affect sampling space (sampling happens in original space)
        return False
    
    def get_step_info(self) -> dict:
        info = super().get_step_info()
        info['n_components'] = self.n_components
        info['kernel'] = self.kernel
        if self.gamma is not None:
            info['gamma'] = self.gamma
        return info
