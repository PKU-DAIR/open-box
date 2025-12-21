from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace, Configuration
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import TransformativeProjectionStep
from ...core import OptimizerProgress


class QuantizationProjectionStep(TransformativeProjectionStep):
    def __init__(self, 
                 method: str = 'quantization',
                 max_num_values: int = 10,
                 seed: int = 42, adaptive: bool = False,
                 **kwargs):
        super().__init__(method=method, **kwargs)
        self._max_num_values = max_num_values
        self.seed = seed
        self._rs = np.random.RandomState(seed=seed)
        
        self._knobs_scalers: dict = {}
        self.adaptive = adaptive
    
    def _build_projected_space(self, input_space: ConfigurationSpace) -> ConfigurationSpace:
        self._knobs_scalers = {}
        root_hyperparams = []
        quantized_params = []
        unchanged_params = []
        
        for adaptee_hp in input_space.get_hyperparameters():
            if not isinstance(adaptee_hp, CSH.UniformIntegerHyperparameter):
                root_hyperparams.append(adaptee_hp)
                unchanged_params.append(adaptee_hp.name)
                continue
            
            if not self._needs_quantization(adaptee_hp):
                root_hyperparams.append(adaptee_hp)
                unchanged_params.append(adaptee_hp.name)
                continue
            
            # Quantize knob
            # original value: [lower, upper] => quantized value: [1, max_num_values]
            lower, upper = adaptee_hp.lower, adaptee_hp.upper
            scaler = MinMaxScaler(feature_range=(lower, upper))
            scaler.fit([[1], [self._max_num_values]])
            self._knobs_scalers[adaptee_hp.name] = scaler
            
            default_value = round(
                scaler.inverse_transform([[adaptee_hp.default_value]])[0][0]
            )
            default_value = max(1, min(self._max_num_values, default_value))
            
            quantized_hp = CSH.UniformIntegerHyperparameter(
                f'{adaptee_hp.name}|q', 1, self._max_num_values,
                default_value=default_value,
            )
            root_hyperparams.append(quantized_hp)
            
            # 记录量化信息
            original_num_values = upper - lower + 1
            compression_ratio = self._max_num_values / original_num_values
            quantized_params.append({
                'name': adaptee_hp.name,
                'type': 'UniformIntegerHyperparameter',
                'original_range': (int(lower), int(upper)),
                'compressed_range': (1, self._max_num_values),
                'original_num_values': original_num_values,
                'quantized_num_values': self._max_num_values,
                'compression_ratio': compression_ratio
            })
        
        root = CS.ConfigurationSpace(
            name=input_space.name,
            seed=self.seed,
        )
        root.add_hyperparameters(root_hyperparams)
        
        if quantized_params:
            avg_compression_ratio = sum(p['compression_ratio'] for p in quantized_params) / len(quantized_params)
        else:
            avg_compression_ratio = 1.0
        
        self.compression_info = {
            'compressed_params': quantized_params,
            'unchanged_params': unchanged_params,
            'total_quantized': len(quantized_params),
            'avg_compression_ratio': avg_compression_ratio
        }
        
        return root
    
    def _needs_quantization(self, hp: CSH.UniformIntegerHyperparameter) -> bool:
        return (hp.upper - hp.lower + 1) > self._max_num_values
    
    def unproject_point(self, point: Configuration) -> dict:
        coords = point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)
        valid_dim_names = [dim.name for dim in self.input_space.get_hyperparameters()]
        unproject_coords = {}
        
        for name, value in coords.items():
            dequantize = name.endswith('|q')
            if not dequantize:
                unproject_coords[name] = value
                continue
            
            # De-quantize
            dim_name = name[:-2]
            if dim_name not in valid_dim_names or dim_name not in self._knobs_scalers:
                logger.warning(f"Cannot dequantize {name}, keeping original value")
                unproject_coords[name] = value
                continue
            
            scaler = self._knobs_scalers[dim_name]
            lower, upper = scaler.feature_range
            
            value = int(scaler.transform([[value]])[0][0])
            value = max(lower, min(upper, value))
            unproject_coords[dim_name] = value
        
        return unproject_coords
    
    def project_point(self, point) -> dict:
        if isinstance(point, Configuration):
            original_dict = point.get_dictionary()
        elif isinstance(point, dict):
            original_dict = point
        else:
            original_dict = dict(point)
        
        quantized_dict = {}
        
        for name, value in original_dict.items():
            if name in self._knobs_scalers:
                scaler = self._knobs_scalers[name]
                # Use inverse_transform:
                # original value [lower, upper] -> quantized value [1, max_num_values]
                # The scaler maps [1, max_num_values] -> [lower, upper],
                # so inverse maps [lower, upper] -> [1, max_num_values]
                lower, upper = scaler.feature_range
                value_clamped = max(lower, min(upper, value))
                quantized_value = round(scaler.inverse_transform([[value_clamped]])[0][0])
                quantized_value = max(1, min(self._max_num_values, quantized_value))
                quantized_dict[f'{name}|q'] = quantized_value
            else:
                quantized_dict[name] = value
        
        return quantized_dict
    
    def supports_adaptive_update(self) -> bool:
        return self.adaptive
    
    def update(self, progress: OptimizerProgress, history: History) -> bool:
        # Stagnant: reduce quantization (increase max_num_values) to expand search space
        if progress.is_stagnant(threshold=5):
            old_max = self._max_num_values
            self._max_num_values = min(100, self._max_num_values + 5)  # Increase by 5, cap at 100
            if self._max_num_values != old_max:
                logger.info(f"Stagnation detected, increasing quantization factor: {old_max} -> {self._max_num_values}")
                return True
        
        # Improving: can increase quantization (decrease max_num_values) to focus search
        elif progress.has_improvement(threshold=3):
            old_max = self._max_num_values
            self._max_num_values = max(5, self._max_num_values - 2)  # Decrease by 2, floor at 5
            if self._max_num_values != old_max:
                logger.info(f"Improvement detected, decreasing quantization factor: {old_max} -> {self._max_num_values}")
                return True
        
        return False
    
    def get_step_info(self) -> dict:
        info = super().get_step_info()
        info['max_num_values'] = self._max_num_values
        return info