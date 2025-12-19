from abc import ABC, abstractmethod
from typing import Optional
from openbox.utils.history import History
from .progress import OptimizerProgress


class UpdateStrategy(ABC):    
    @abstractmethod
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class PeriodicUpdateStrategy(UpdateStrategy):    
    def __init__(self, period: int = 10):
        self.period = period
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return progress.should_periodic_update(period=self.period)
    
    def get_name(self) -> str:
        return f"periodic(every {self.period} iters)"


class StagnationUpdateStrategy(UpdateStrategy):    
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return progress.is_stagnant(threshold=self.threshold)
    
    def get_name(self) -> str:
        return f"stagnation(threshold={self.threshold})"


class ImprovementUpdateStrategy(UpdateStrategy):    
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return progress.has_improvement(threshold=self.threshold)
    
    def get_name(self) -> str:
        return f"improvement(threshold={self.threshold})"


class CompositeUpdateStrategy(UpdateStrategy):    
    def __init__(self, *strategies: UpdateStrategy):
        self.strategies = strategies
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return any(s.should_update(progress, history) for s in self.strategies)
    
    def get_name(self) -> str:
        names = [s.get_name() for s in self.strategies]
        return f"composite({' OR '.join(names)})"


class HybridUpdateStrategy(UpdateStrategy):    
    def __init__(self, 
                 period: int = 10,
                 stagnation_threshold: Optional[int] = None,
                 improvement_threshold: Optional[int] = None):
        self.period = period
        self.stagnation_threshold = stagnation_threshold
        self.improvement_threshold = improvement_threshold
        
        strategies = [PeriodicUpdateStrategy(period)]
        if stagnation_threshold is not None:
            strategies.append(StagnationUpdateStrategy(stagnation_threshold))
        if improvement_threshold is not None:
            strategies.append(ImprovementUpdateStrategy(improvement_threshold))
        
        self.composite = CompositeUpdateStrategy(*strategies)
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return self.composite.should_update(progress, history)
    
    def get_name(self) -> str:
        parts = [f"periodic({self.period})"]
        if self.stagnation_threshold is not None:
            parts.append(f"stagnant({self.stagnation_threshold})")
        if self.improvement_threshold is not None:
            parts.append(f"improve({self.improvement_threshold})")
        return " OR ".join(parts)

