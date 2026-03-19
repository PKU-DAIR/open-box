import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from .generator import SearchGenerator


class StrategySelector(ABC):    
    @abstractmethod
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        """Select a strategy from the list of strategies
        
        Parameters
        ----------
        strategies : List[SearchGenerator]
            List of strategies to choose from
        iteration : int
            Current iteration number
            
        Returns
        -------
        SearchGenerator
            Selected strategy
        """
        pass
    
    def reset(self):
        pass


class FixedSelector(StrategySelector):
    """Fixed selector
    
    Always select the fixed strategy.
    
    Suitable scenarios: only one strategy, or want to fix using a specific strategy.
    """
    
    def __init__(self, index: int = 0):
        """
        Parameters
        ----------
        index : int, default=0
            Index of the fixed strategy to select
        """
        self.index = index
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        if self.index >= len(strategies):
            raise ValueError(f"Index {self.index} out of range for {len(strategies)} strategies")
        return strategies[self.index]


class ProbabilisticSelector(StrategySelector):
    """Probabilistic selector
        
    Select a strategy from the list of strategies according to the given probability distribution.
    
    Parameters
    ----------
    probabilities : List[float]
        Probability of selecting each strategy, and must sum to 1
    rng : np.random.RandomState, optional
        Random number generator
        
    Examples
    --------
    >>> # 85% probability select local search, 15% probability select random search
    >>> selector = ProbabilisticSelector([0.85, 0.15])
    >>> strategy = selector.select([local_strategy, random_strategy], iteration=0)
    """
    
    def __init__(self, 
                 probabilities: List[float],
                 rng: np.random.RandomState):
        sum_probs = sum(probabilities)
        if abs(sum_probs - 1.0) > 1e-6:
            probabilities = [prob / sum_probs for prob in probabilities]
        
        self.probabilities = np.array(probabilities)
        self.rng = rng
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        if len(strategies) != len(self.probabilities):
            raise ValueError(
                f"Number of strategies ({len(strategies)}) must match "
                f"number of probabilities ({len(self.probabilities)})"
            )
        
        idx = self.rng.choice(len(strategies), p=self.probabilities)
        return strategies[idx]


class InterleavedSelector(StrategySelector):
    """Interleaved selector
    
    Select a strategy from the list of strategies according to the given weights.
    
    Parameters
    ----------
    weights : List[int]
        Weight of each strategy (execution count ratio)
        For example, [4, 1] means 4 times out of 5, select strategy 0, and 1 time out of 5, select strategy 1
        
    Examples
    --------
    >>> # 4 times local search, 1 time random search
    >>> selector = InterleavedSelector([4, 1])
    >>> for i in range(10):
    ...     strategy = selector.select([local, random], i)
    # Result: local, local, local, local, random, local, local, local, local, random
    """
    
    def __init__(self, weights: List[int]):
        if not all(w > 0 for w in weights):
            raise ValueError("All weights must be positive")
        
        self.weights = weights
        self.total = sum(weights)
        self._counter = 0
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        if len(strategies) != len(self.weights):
            raise ValueError(
                f"Number of strategies ({len(strategies)}) must match "
                f"number of weights ({len(self.weights)})"
            )
        
        position = self._counter % self.total
        
        cumsum = 0
        for i, weight in enumerate(self.weights):
            cumsum += weight
            if position < cumsum:
                self._counter += 1
                return strategies[i]
        
        self._counter += 1
        return strategies[0]
    
    def reset(self):
        self._counter = 0


class RoundRobinSelector(StrategySelector):
    """RoundRobin selector
    
    Select a strategy from the list of strategies in a round-robin manner (each strategy has the same weight).
    
    Examples
    --------
    >>> selector = RoundRobinSelector()
    >>> for i in range(6):
    ...     strategy = selector.select([s1, s2, s3], i)
    # Result: s1, s2, s3, s1, s2, s3
    """
    
    def __init__(self):
        self._counter = 0
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        idx = self._counter % len(strategies)
        self._counter += 1
        return strategies[idx]
    
    def reset(self):
        self._counter = 0


class AdaptiveSelector(StrategySelector):
    """Adaptive selector
    
    Dynamically adjust the selection probabilities of strategies based on their historical performance.
    Strategies with better performance will get higher selection probabilities.
    
    Parameters
    ----------
    initial_probs : List[float]
        Initial probability distribution
    learning_rate : float, default=0.1
        Learning rate, control the speed of probability update
    temperature : float, default=1.0
        Temperature parameter, control the balance between exploration and exploitation
        - Temperature high: more uniform probabilities (more exploration)
        - Temperature low: more concentrated probabilities (more exploitation)
    rng : np.random.RandomState
        Random number generator
        
    Notes
    -----
    Need to provide feedback through the update() method to adjust the probabilities.
    
    Examples
    --------
    >>> selector = AdaptiveSelector([0.5, 0.5])
    >>> 
    >>> # Use the strategy and get the performance metric
    >>> strategy = selector.select([s1, s2], iteration=0)
    >>> improvement = evaluate_strategy_performance(strategy)
    >>> 
    >>> # Update the probability (the larger the improvement, the better)
    >>> selector.update(strategy_index=0, reward=improvement)
    """
    
    def __init__(self,
                 initial_probs: List[float],
                 learning_rate: float = 0.1,
                 temperature: float = 1.0,
                 rng: np.random.RandomState = None):
        sum_probs = sum(initial_probs)
        if (abs(sum_probs) - 1.0) > 1e-6:
            initial_probs = [prob / sum_probs for prob in initial_probs]
        
        self.probs = np.array(initial_probs, dtype=float)
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.rng = rng
        
        self.rewards = np.zeros(len(initial_probs))
        self.counts = np.zeros(len(initial_probs))
        
        self._last_selected = None
    
    def select(self, strategies: List[SearchGenerator], iteration: int) -> SearchGenerator:
        if len(strategies) != len(self.probs):
            raise ValueError(
                f"Number of strategies ({len(strategies)}) must match "
                f"number of probabilities ({len(self.probs)})"
            )
        
        temp_probs = self._apply_temperature(self.probs)
        
        idx = self.rng.choice(len(strategies), p=temp_probs)
        self._last_selected = idx
        self.counts[idx] += 1
        
        return strategies[idx]
    
    def update(self, strategy_index: int, reward: float):
        """Update the probability of a strategy
        
        Parameters
        ----------
        strategy_index : int
            Index of the strategy
        reward : float
            Reward value (the larger the better)
            For example, can use improvement = old_best - new_best
        """
        self.rewards[strategy_index] += reward
        
        avg_reward = self.rewards[strategy_index] / max(self.counts[strategy_index], 1)
        
        # Simple update rule: strategies with better performance increase probability
        self.probs[strategy_index] += self.learning_rate * avg_reward
        
        self.probs = np.maximum(self.probs, 0.01)  # keep minimum probability
        self.probs = self.probs / self.probs.sum()
    
    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        if self.temperature == 1.0:
            return probs
        
        # softmax with temperature, scale the probabilities
        log_probs = np.log(probs + 1e-10)
        scaled = log_probs / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / exp_scaled.sum()
    
    def reset(self):
        self.probs = np.ones(len(self.probs)) / len(self.probs)
        self.rewards = np.zeros(len(self.probs))
        self.counts = np.zeros(len(self.probs))
        self._last_selected = None
    
    def get_statistics(self) -> dict:
        return {
            'probabilities': self.probs.tolist(),
            'avg_rewards': (self.rewards / np.maximum(self.counts, 1)).tolist(),
            'counts': self.counts.tolist()
        }

