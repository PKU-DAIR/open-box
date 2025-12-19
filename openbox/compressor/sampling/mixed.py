import random
from typing import List, Optional, Tuple, Dict, Any
from openbox import logger
from ConfigSpace import ConfigurationSpace, Configuration
from .base import SamplingStrategy


class MixedRangeSamplingStrategy(SamplingStrategy):    
    def __init__(self, 
                 compressed_space: ConfigurationSpace,
                 original_space: ConfigurationSpace,
                 initial_prob: float = 0.9,
                 method: str = 'boundary',
                 seed: Optional[int] = None):
        """
        Args:
            compressed_space: Compressed configuration space
            original_space: Original configuration space
            initial_prob: Initial probability of sampling from compressed space
            method: Compression method ('boundary' or 'expert')
            seed: Random seed
        """
        self.compressed_space = compressed_space
        self.original_space = original_space
        self.compressed_prob = initial_prob
        self.method = method
        self.seed = seed
        
        if seed is not None:
            self.compressed_space.seed(seed)
            self.original_space.seed(seed)
            random.seed(seed)
        
        self.compressed_results: List[float] = []
        self.original_results: List[float] = []
    
    def sample(self, n: int = 1) -> List[Configuration]:
        configs = []
        for _ in range(n):
            if random.random() < self.compressed_prob:
                config = self.compressed_space.sample_configuration()
                config._sampled_from = 'compressed'
            else:
                config = self.original_space.sample_configuration()
                config._sampled_from = 'original'
            configs.append(config)
        return configs
    
    def update_probabilities(self, results: List[Tuple[Configuration, float]]):
        compressed_perf = []
        original_perf = []
        
        for config, perf in results:
            if hasattr(config, '_sampled_from'):
                if config._sampled_from == 'compressed':
                    compressed_perf.append(perf)
                elif config._sampled_from == 'original':
                    original_perf.append(perf)
        
        self.compressed_results.extend(compressed_perf)
        self.original_results.extend(original_perf)
        
        if len(compressed_perf) > 0 and len(original_perf) > 0:
            compressed_mean = sum(compressed_perf) / len(compressed_perf)
            original_mean = sum(original_perf) / len(original_perf)
            
            if original_mean < compressed_mean:
                self.compressed_prob = max(0.5, self.compressed_prob - 0.1)
                logger.info(f"Original range performing better. Adjusting compressed_prob to {self.compressed_prob:.2f}")
            else:
                self.compressed_prob = min(0.95, self.compressed_prob + 0.05)
                logger.info(f"Compressed range performing better. Adjusting compressed_prob to {self.compressed_prob:.2f}")
    
    def get_spaces(self) -> Tuple[ConfigurationSpace, Optional[ConfigurationSpace]]:
        return (self.compressed_space, self.original_space)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'compressed_prob': self.compressed_prob,
            'compressed_samples': len(self.compressed_results),
            'original_samples': len(self.original_results),
            'compressed_mean': sum(self.compressed_results) / len(self.compressed_results) if self.compressed_results else None,
            'original_mean': sum(self.original_results) / len(self.original_results) if self.original_results else None,
        }

