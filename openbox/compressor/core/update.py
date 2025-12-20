from abc import ABC, abstractmethod
from typing import Optional, Tuple
from openbox.utils.history import History
from .progress import OptimizerProgress


class UpdateStrategy(ABC):    
    @abstractmethod
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        pass
    
    @abstractmethod
    def compute_new_topk(self, 
                        current_topk: int,
                        reduction_ratio: float,
                        min_dimensions: int,
                        max_dimensions: Optional[int],
                        progress: OptimizerProgress) -> Tuple[int, str]:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class PeriodicUpdateStrategy(UpdateStrategy):    
    def __init__(self, period: int = 10):
        self.period = period
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return progress.should_periodic_update(period=self.period)
    
    def compute_new_topk(self, 
                        current_topk: int,
                        reduction_ratio: float,
                        min_dimensions: int,
                        max_dimensions: Optional[int],
                        progress: OptimizerProgress) -> Tuple[int, str]:
        reduction = int(current_topk * reduction_ratio)
        new_topk = max(min_dimensions, current_topk - reduction)
        description = f"Periodic update (iteration {progress.iteration}): reducing dimensions {current_topk} -> {new_topk}"
        return new_topk, description
    
    def get_name(self) -> str:
        return f"periodic(every {self.period} iters)"


class StagnationUpdateStrategy(UpdateStrategy):    
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return progress.is_stagnant(threshold=self.threshold)
    
    def compute_new_topk(self, 
                        current_topk: int,
                        reduction_ratio: float,
                        min_dimensions: int,
                        max_dimensions: Optional[int],
                        progress: OptimizerProgress) -> Tuple[int, str]:
        increase = int(current_topk * reduction_ratio)
        new_topk = current_topk + increase
        if max_dimensions is not None:
            new_topk = min(new_topk, max_dimensions)
        description = f"Stagnation detected, increasing dimensions: {current_topk} -> {new_topk}"
        return new_topk, description
    
    def get_name(self) -> str:
        return f"stagnation(threshold={self.threshold})"


class ImprovementUpdateStrategy(UpdateStrategy):    
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return progress.has_improvement(threshold=self.threshold)
    
    def compute_new_topk(self, 
                        current_topk: int,
                        reduction_ratio: float,
                        min_dimensions: int,
                        max_dimensions: Optional[int],
                        progress: OptimizerProgress) -> Tuple[int, str]:
        reduction = int(current_topk * reduction_ratio)
        new_topk = max(min_dimensions, current_topk - reduction)
        description = f"Improvement detected, reducing dimensions: {current_topk} -> {new_topk}"
        return new_topk, description
    
    def get_name(self) -> str:
        return f"improvement(threshold={self.threshold})"


class CompositeUpdateStrategy(UpdateStrategy):    
    def __init__(self, *strategies: UpdateStrategy):
        self.strategies = strategies
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return any(s.should_update(progress, history) for s in self.strategies)
    
    def compute_new_topk(self, 
                        current_topk: int,
                        reduction_ratio: float,
                        min_dimensions: int,
                        max_dimensions: Optional[int],
                        progress: OptimizerProgress) -> Tuple[int, str]:
        # check each strategy in order
        for strategy in self.strategies:
            if strategy.should_update(progress, None):
                return strategy.compute_new_topk(
                    current_topk, reduction_ratio, min_dimensions, max_dimensions, progress
                )
        return current_topk, "No update triggered"
    
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
        
        self.periodic_strategy = PeriodicUpdateStrategy(period)
        self.stagnation_strategy = StagnationUpdateStrategy(stagnation_threshold) if stagnation_threshold is not None else None
        self.improvement_strategy = ImprovementUpdateStrategy(improvement_threshold) if improvement_threshold is not None else None
        
        self.strategies = [self.periodic_strategy]
        if self.stagnation_strategy:
            self.strategies.append(self.stagnation_strategy)
        if self.improvement_strategy:
            self.strategies.append(self.improvement_strategy)
    
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        return any(s.should_update(progress, history) for s in self.strategies)
    
    def compute_new_topk(self, 
                        current_topk: int,
                        reduction_ratio: float,
                        min_dimensions: int,
                        max_dimensions: Optional[int],
                        progress: OptimizerProgress) -> Tuple[int, str]:
        # Priority: stagnation => improvement => periodic
        # Check stagnation first (highest priority)
        if self.stagnation_strategy and self.stagnation_strategy.should_update(progress, None):
            return self.stagnation_strategy.compute_new_topk(
                current_topk, reduction_ratio, min_dimensions, max_dimensions, progress
            )
        if self.improvement_strategy and self.improvement_strategy.should_update(progress, None):
            return self.improvement_strategy.compute_new_topk(
                current_topk, reduction_ratio, min_dimensions, max_dimensions, progress
            )
        return self.periodic_strategy.compute_new_topk(
            current_topk, reduction_ratio, min_dimensions, max_dimensions, progress
        )
    
    def get_name(self) -> str:
        parts = [f"periodic({self.period})"]
        if self.stagnation_threshold is not None:
            parts.append(f"stagnant({self.stagnation_threshold})")
        if self.improvement_threshold is not None:
            parts.append(f"improve({self.improvement_threshold})")
        return " OR ".join(parts)

