from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ConfigSpace import ConfigurationSpace, Configuration


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, n: int = 1) -> List[Configuration]:
        pass
    
    def update_probabilities(self, results: List[Tuple[Configuration, float]]):
        pass
    
    @abstractmethod
    def get_spaces(self) -> Tuple[ConfigurationSpace, Optional[ConfigurationSpace]]:
        pass

