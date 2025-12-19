import random
from typing import List, Optional, Tuple
from ConfigSpace import ConfigurationSpace, Configuration
from .base import SamplingStrategy


class StandardSamplingStrategy(SamplingStrategy):    
    def __init__(self, space: ConfigurationSpace, seed: Optional[int] = None):
        self.space = space
        self.seed = seed
        if seed is not None:
            self.space.seed(seed)
            random.seed(seed)
    
    def sample(self, n: int = 1) -> List[Configuration]:
        configs = []
        for _ in range(n):
            config = self.space.sample_configuration()
            configs.append(config)
        return configs
    
    def get_spaces(self) -> Tuple[ConfigurationSpace, Optional[ConfigurationSpace]]:
        return (self.space, None)

