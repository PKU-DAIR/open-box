import copy
from typing import Optional, List, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from ...core.step import CompressionStep


class RangeCompressionStep(CompressionStep):    
    def __init__(self, method: str = 'boundary', **kwargs):
        super().__init__('range_compression', **kwargs)
        self.method = method
        self.original_space: Optional[ConfigurationSpace] = None
    
    def compress(self, input_space: ConfigurationSpace, 
                space_history: Optional[List[History]] = None,
                source_similarities: Optional[Dict[int, float]] = None,
                **kwargs) -> ConfigurationSpace:
        if self.method == 'none':
            logger.info("Range compression disabled, returning input space")
            return input_space
        
        self.original_space = copy.deepcopy(input_space)
        compressed_space = self._compute_compressed_space(input_space, space_history, source_similarities)
        
        self.compression_info = self._collect_compression_details(input_space, compressed_space)

        logger.info(f"Range compression: {len(input_space.get_hyperparameters())} parameters compressed")
        return compressed_space
    
    def _collect_compression_details(self, input_space: ConfigurationSpace, 
                                    compressed_space: ConfigurationSpace) -> dict:
        details = {
            'compressed_params': [],
            'unchanged_params': [],
            'avg_compression_ratio': 1.0
        }
        
        for hp in input_space.get_hyperparameters():
            name = hp.name
            if name not in [h.name for h in compressed_space.get_hyperparameters()]:
                continue
            
            compressed_hp = compressed_space.get_hyperparameter(name)
            
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                original_range = (float(hp.lower), float(hp.upper))
                compressed_range = (float(compressed_hp.lower), float(compressed_hp.upper))
                
                if abs(original_range[0] - compressed_range[0]) > 1e-6 or abs(original_range[1] - compressed_range[1]) > 1e-6:
                    compression_ratio = (compressed_range[1] - compressed_range[0]) / (original_range[1] - original_range[0])
                    details['compressed_params'].append({
                        'name': name,
                        'type': type(hp).__name__,
                        'original_range': original_range,
                        'compressed_range': compressed_range,
                        'compression_ratio': compression_ratio
                    })
                else:
                    details['unchanged_params'].append(name)
            elif hasattr(hp, 'choices'):
                original_choices = list(hp.choices)
                compressed_choices = list(compressed_hp.choices)
                
                if original_choices != compressed_choices:
                    details['compressed_params'].append({
                        'name': name,
                        'type': 'Categorical',
                        'original_choices': original_choices,
                        'compressed_choices': compressed_choices,
                        'compression_ratio': len(compressed_choices) / len(original_choices)
                    })
                else:
                    details['unchanged_params'].append(name)
        
        if details['compressed_params']:
            details['avg_compression_ratio'] = sum(p['compression_ratio'] for p in details['compressed_params']) / len(details['compressed_params'])
        else:
            details['avg_compression_ratio'] = 1.0
        
        return details
    
    def _compute_compressed_space(self, 
                                  input_space: ConfigurationSpace,
                                  space_history: Optional[List[History]] = None,
                                  source_similarities: Optional[Dict[int, float]] = None,
                                  **kwargs) -> ConfigurationSpace:
        return copy.deepcopy(input_space)
    
    def project_point(self, point) -> dict:
        # project a point from input_space to output_space.
        # clip values to compressed ranges and fill missing parameters.
        if hasattr(point, 'get_dictionary'):
            point_dict = point.get_dictionary()
        elif isinstance(point, dict):
            point_dict = point
        else:
            point_dict = dict(point)
        
        # if no compression was applied (output_space not set), return as is
        if self.output_space is None:
            return point_dict
        
        # clip values to the compressed ranges
        from ...filling import clip_values_to_space
        clipped_dict = clip_values_to_space(point_dict, self.output_space, report=False)
        
        # fill missing parameters if needed
        if self.filling_strategy is not None:
            clipped_dict = self.filling_strategy.fill_missing_parameters(
                clipped_dict, self.output_space
            )
        
        return clipped_dict
    
    def needs_unproject(self) -> bool:
        return False
    
    def affects_sampling_space(self) -> bool:
        return True

