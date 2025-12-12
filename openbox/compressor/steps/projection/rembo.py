import numpy as np
from typing import Optional, List
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace, Configuration
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from sklearn.preprocessing import MinMaxScaler

from .base import TransformativeProjectionStep


class REMBOProjectionStep(TransformativeProjectionStep):    
    def __init__(self, 
                 method: str = 'rembo',
                 low_dim: int = 10,
                 max_num_values: Optional[int] = None,
                 seed: int = 42,
                 **kwargs):
        super().__init__(method=method, **kwargs)
        self.low_dim = low_dim
        self.max_num_values = max_num_values
        self.seed = seed
        self._rs = np.random.RandomState(seed=seed)
        
        # Projection matrix and scalers
        self._A: Optional[np.ndarray] = None
        self._scaler: Optional[MinMaxScaler] = None
        self._q_scaler: Optional[MinMaxScaler] = None
        self.active_hps: List = []
        
        self._low_to_high_cache: dict = {}
        self._high_to_low_cache: dict = {}
    
    def _build_projected_space(self, input_space: ConfigurationSpace) -> ConfigurationSpace:
        self.active_hps = list(input_space.get_hyperparameters())
        
        # Create lower dimensionality configuration space
        # Space bounds are [-sqrt(low_dim), sqrt(low_dim)] rather than [-1, 1]
        box_bound = np.sqrt(self.low_dim)
        target = CS.ConfigurationSpace(
            name=input_space.name,
            seed=self.seed
        )
        
        if self._max_num_values is None:
            # Continuous low-dimensional space
            hps = [
                CS.UniformFloatHyperparameter(
                    name=f'rembo_{idx}',
                    lower=-box_bound,
                    upper=box_bound
                )
                for idx in range(self.low_dim)
            ]
            self._q_scaler = None
        else:
            # Quantized low-dimensional space
            logger.info(f'Using quantization: q={self._max_num_values}')
            hps = [
                CS.UniformIntegerHyperparameter(
                    name=f'rembo_{idx}',
                    lower=1,
                    upper=self._max_num_values
                )
                for idx in range(self.low_dim)
            ]
            # (1, q) -> (-sqrt(low_dim), sqrt(low_dim)) scaling
            self._q_scaler = MinMaxScaler(feature_range=(-box_bound, box_bound))
            ones = np.ones(shape=self.low_dim)
            self._q_scaler.fit([ones, ones * self._max_num_values])
        
        target.add_hyperparameters(hps)
        self.output_space = target
        
        # (-sqrt, sqrt) -> (0, 1) scaling
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        bbound_vector = np.ones(len(self.active_hps)) * box_bound
        # Use two points (minimum & maximum)
        self._scaler.fit(
            np.array([-bbound_vector, bbound_vector])
        )
        
        self._A = self._rs.normal(
            0, 1, (len(self.active_hps), self.low_dim)
        )
        
        return target
    
    def unproject_point(self, point: Configuration) -> dict:
        low_dim_point_raw = np.array([
            point.get(f'rembo_{idx}') for idx in range(self.low_dim)
        ])
        
        low_dim_key = tuple(low_dim_point_raw.tolist())
        if low_dim_key in self._low_to_high_cache:
            return self._low_to_high_cache[low_dim_key].copy()

        # Dequantize if needed
        if self._max_num_values is not None:
            assert self._q_scaler is not None
            # Dequantize: (1, q) -> (-sqrt(low_dim), sqrt(low_dim))
            low_dim_point = self._q_scaler.transform([low_dim_point_raw])[0]
        else:
            low_dim_point = low_dim_point_raw
        
        # Project: (-sqrt(low_dim), sqrt(low_dim)) -> (0, 1)
        high_dim_point = [
            np.dot(self._A[idx, :], low_dim_point)
            for idx in range(len(self.active_hps))
        ]
        high_dim_point = self._scaler.transform([high_dim_point])[0]
        
        # Transform back to original space
        high_dim_conf = {}
        dims_clipped = 0
        for hp, value in zip(self.active_hps, high_dim_point):
            if value <= 0 or value >= 1:
                logger.warning(f'Point clipped in dim: {hp.name}')
                dims_clipped += 1
            # Clip value to [0, 1]
            value = np.clip(value, 0., 1.)
            
            if isinstance(hp, CS.CategoricalHyperparameter):
                index = int(value * len(hp.choices))
                index = max(0, min(len(hp.choices) - 1, index))
                value = hp.choices[index]
            elif isinstance(hp, CS.hyperparameters.NumericalHyperparameter):
                value = hp._transform(value)
                value = max(hp.lower, min(hp.upper, value))
            else:
                raise NotImplementedError(f"Unsupported hyperparameter type: {type(hp)}")
            
            high_dim_conf[hp.name] = value
        
        if dims_clipped > 0:
            logger.warning(f'Clipped {dims_clipped} dimensions during unprojection')
        
        self._low_to_high_cache[low_dim_key] = high_dim_conf.copy()
        high_dim_key = tuple(sorted(high_dim_conf.items()))
        self._high_to_low_cache[high_dim_key] = low_dim_key
        
        return high_dim_conf
    
    def project_point(self, point) -> dict:
        if isinstance(point, Configuration):
            high_dim_dict = point.get_dictionary()
        elif isinstance(point, dict):
            high_dim_dict = point
        else:
            high_dim_dict = dict(point)
        
        high_dim_key = tuple(sorted(high_dim_dict.items()))
        if high_dim_key in self._high_to_low_cache:
            low_dim_key = self._high_to_low_cache[high_dim_key]
            low_dim_point = np.array(low_dim_key)
            return {f'rembo_{idx}': float(low_dim_point[idx]) for idx in range(self.low_dim)}
        else:
            logger.error(f"Cache miss in project_point for high-dim point: {high_dim_dict}")
            raise ValueError(f"Cache miss in project_point for high-dim point: {high_dim_dict}")