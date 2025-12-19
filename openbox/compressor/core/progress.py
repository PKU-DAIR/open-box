from typing import List, Optional
from openbox.utils.history import History


class OptimizerProgress:
    def __init__(self):
        self.iteration = 0
        self.best_value_history: List[float] = []
        self.improvement_count = 0
        self.stagnation_count = 0
        self.last_best_value: Optional[float] = None
        self.minimize = True  # True for minimization, False for maximization
    
    def update(self, current_best_value: float, minimize: bool = True):
        self.iteration += 1
        self.minimize = minimize
        
        if self.last_best_value is not None:
            if minimize:
                improved = current_best_value < self.last_best_value
            else:
                improved = current_best_value > self.last_best_value
            
            if improved:
                self.improvement_count += 1
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
                self.improvement_count = 0
        else:
            # First iteration
            self.improvement_count = 0
            self.stagnation_count = 0
        
        self.last_best_value = current_best_value
        self.best_value_history.append(current_best_value)
    
    def update_from_history(self, history: History):
        if history is None or len(history) == 0:
            return
        
        incumbent_value = history.get_incumbent_value()
        if incumbent_value is not None:
            minimize = True  # Default, should be inferred from history if possible
            self.update(incumbent_value, minimize=minimize)
    
    def has_improvement(self, threshold: int = 3) -> bool:
        return self.improvement_count >= threshold
    
    def is_stagnant(self, threshold: int = 5) -> bool:
        return self.stagnation_count >= threshold
    
    def should_periodic_update(self, period: int = 10) -> bool:
        return self.iteration > 0 and self.iteration % period == 0
    
    def get_recent_trend(self, window: int = 5) -> str:
        if len(self.best_value_history) < window:
            return 'stable'
        
        recent = self.best_value_history[-window:]
        if self.minimize:
            if recent[-1] < recent[0]:
                return 'improving'
            elif recent[-1] > recent[0]:
                return 'degrading'
        else:
            if recent[-1] > recent[0]:
                return 'improving'
            elif recent[-1] < recent[0]:
                return 'degrading'
        
        return 'stable'
    
    def reset(self):
        self.iteration = 0
        self.best_value_history = []
        self.improvement_count = 0
        self.stagnation_count = 0
        self.last_best_value = None

