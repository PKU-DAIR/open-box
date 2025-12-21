"""
Expert-specified range compression step.
"""

import copy
from typing import Optional, List, Dict, Tuple
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .base import RangeCompressionStep
from ...sampling import MixedRangeSamplingStrategy
from ...utils import create_space_from_ranges


class ExpertRangeStep(RangeCompressionStep):    
    def __init__(self, 
                 method: str = 'expert',
                 expert_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 enable_mixed_sampling: bool = False,
                 initial_prob: float = 0.9,
                 seed: Optional[int] = None,
                 **kwargs):
        super().__init__(method=method, **kwargs)

        self.expert_ranges = expert_ranges or {}
        self.enable_mixed_sampling = enable_mixed_sampling
        self.initial_prob = initial_prob
        self.seed = seed
    
    def _compute_compressed_space(self, 
                                  input_space: ConfigurationSpace,
                                  space_history: Optional[List[History]] = None,
                                  source_similarities: Optional[Dict[int, float]] = None) -> ConfigurationSpace:
        if not self.expert_ranges:
            logger.warning("No expert ranges provided, returning input space")
            return copy.deepcopy(input_space)
        
        valid_ranges = {}
        param_names = input_space.get_hyperparameter_names()
        
        for param_name, (min_val, max_val) in self.expert_ranges.items():
            if param_name not in param_names:
                logger.warning(f"Expert parameter '{param_name}' not found in configuration space")
                continue

            if min_val >= max_val:
                logger.warning(f"Invalid expert range [{min_val}, {max_val}] for {param_name}, skipping")
                continue
            
            hp = input_space.get_hyperparameter(param_name)
            if not (hasattr(hp, 'lower') and hasattr(hp, 'upper')):
                logger.warning(f"Parameter '{param_name}' is not numeric, skipping")
                continue
            
            original_min = hp.lower
            original_max = hp.upper
            min_val = max(min_val, original_min)
            max_val = min(max_val, original_max)
            
            if min_val >= max_val:
                logger.warning(f"Expert range for {param_name} is invalid after clamping, skipping")
                continue
            valid_ranges[param_name] = (min_val, max_val)
        
        if not valid_ranges:
            logger.warning("No valid expert ranges, returning input space")
            return copy.deepcopy(input_space)
        
        compressed_space = create_space_from_ranges(input_space, valid_ranges)
        logger.info(f"Expert range compression: {len(valid_ranges)} parameters compressed")
        logger.info(f"Compressed parameters: {list(valid_ranges.keys())}")
        return compressed_space
    
    def get_sampling_strategy(self):
        if self.enable_mixed_sampling and self.original_space is not None:
            compressed_space = self.output_space if self.output_space else self.original_space
            return MixedRangeSamplingStrategy(
                compressed_space=compressed_space,
                original_space=self.original_space,
                initial_prob=self.initial_prob,
                method='expert',
                seed=self.seed
            )
        return None

    def get_step_info(self) -> dict:
        info = super().get_step_info()
        info['n_expert_ranges'] = len(self.expert_ranges)
        info['expert_ranges'] = self.expert_ranges
        info['enable_mixed_sampling'] = self.enable_mixed_sampling
        info['initial_prob'] = self.initial_prob
        info['seed'] = self.seed
        return info
