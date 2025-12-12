import numpy as np
from typing import Optional, List
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace, Configuration
import ConfigSpace as CS
from sklearn.preprocessing import MinMaxScaler

from .base import TransformativeProjectionStep


class HesBOProjectionStep(TransformativeProjectionStep):    
    def __init__(self, 
                 method: str = 'hesbo',
                 low_dim: int = 10,
                 max_num_values: Optional[int] = None,
                 seed: int = 42,
                 **kwargs):
        super().__init__(method=method, **kwargs)
        self.low_dim = low_dim
        self._max_num_values = max_num_values
        self.seed = seed
        self._rs = np.random.RandomState(seed=seed)
        
        self._scaler: Optional[MinMaxScaler] = None
        self._h: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self.active_hps: List = []
    
    def _build_projected_space(self, input_space: ConfigurationSpace) -> ConfigurationSpace:
        self.active_hps = list(input_space.get_hyperparameters())
        
        target = CS.ConfigurationSpace(
            name=input_space.name,
            seed=self.seed
        )
        
        if self._max_num_values is None:    # no quantization
            hps = [
                CS.UniformFloatHyperparameter(
                    name=f'hesbo_{idx}',
                    lower=-1,
                    upper=1
                )
                for idx in range(self.low_dim)
            ]
        else:
            # Use quantization, step size: 2. / max_num_values
            logger.info(f'Using quantization: q={self._max_num_values}')
            q = 2. / self._max_num_values
            hps = [
                CS.UniformFloatHyperparameter(
                    name=f'hesbo_{idx}',
                    lower=-1,
                    upper=1,
                    q=q
                )
                for idx in range(self.low_dim)
            ]
        
        target.add_hyperparameters(hps)
        self.output_space = target
        
        # (-1, 1) -> (0, 1) scaling
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        ones = np.ones(len(self.active_hps))
        # Use two points (minimum & maximum)
        self._scaler.fit(
            np.array([-ones, ones])
        )
        
        # Implicitly define matrix S' using hashing
        # _h: maps each high-dim index to a low-dim index
        # _sigma: sign for each high-dim dimension
        self._h = self._rs.choice(
            range(self.low_dim), len(self.active_hps)
        )
        self._sigma = self._rs.choice([-1, 1], len(self.active_hps))
        
        return target
    
    def unproject_point(self, point: Configuration) -> dict:
        low_dim_point = [
            point.get(f'hesbo_{idx}') for idx in range(self.low_dim)
        ]
        
        # Project using hashing: high_dim[i] = sigma[i] * low_dim[h[i]]
        high_dim_point = [
            self._sigma[idx] * low_dim_point[self._h[idx]]
            for idx in range(len(self.active_hps))
        ]
        high_dim_point = self._scaler.transform([high_dim_point])[0]
        
        # Transform back to original space
        high_dim_conf = {}
        for hp, value in zip(self.active_hps, high_dim_point):
            # HesBO does not project values outside of range
            # NOTE: need this cause of weird floating point errors
            value = max(0, min(1, value))
            
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
        
        return high_dim_conf
    
    def project_point(self, point) -> dict:
        # HesBO projection is many-to-one, so we use an approximation
        # Sample a random point in low-dim space that could map to this point
        if self._max_num_values is None:
            # Continuous: sample uniformly from [-1, 1]
            low_dim_point = self._rs.uniform(-1, 1, size=self.low_dim)
        else:
            # Quantized: sample uniformly with step size
            q = 2. / self._max_num_values
            num_steps = int(2 / q) + 1
            low_dim_point = self._rs.choice(
                np.arange(-1, 1 + q, q), size=self.low_dim
            )
        
        return {f'hesbo_{idx}': float(low_dim_point[idx]) for idx in range(self.low_dim)}
