import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from ConfigSpace import ConfigurationSpace, Configuration
from openbox import logger
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from .generator import SearchGenerator, LocalSearchGenerator,CMAESGenerator
from .utils import convert_configurations_to_array
from .selector import StrategySelector, FixedSelector
from openbox.utils.history import Observation, History

class AcquisitionOptimizer(ABC):
    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        config_space: ConfigurationSpace,
        rng: np.random.RandomState = np.random.RandomState(42)
    ):
        self.acq = acquisition_function
        self.config_space = config_space
        
        self.rng = rng
        self.iter_id = 0
    
    @abstractmethod
    def _maximize(self, history:History, num_points: int, excluded_configs: List[Configuration] = [], **kwargs) -> List[Tuple]:
        pass
    
    def maximize(self, history:History, num_points: int, excluded_configs: List[Configuration] = [], **kwargs) -> List:
        results = self._maximize(history, num_points, excluded_configs, **kwargs)
        return [result[1] for result in results]
    
    def _evaluate_batch(self, configs: List[Configuration], **kwargs) -> np.ndarray:
        return self._acquisition_function(configs, **kwargs).flatten()
    
    def _sort_configs_by_acq_value(self, configs, **kwargs):
        acq_values = self._acquisition_function(configs, **kwargs).flatten()
        random_values = self.rng.rand(len(acq_values))
        # Sort by acquisition value (primary) and random tie-breaker (secondary)
        # Last column is primary sort key
        indices = np.lexsort((random_values.flatten(), acq_values.flatten()))
        return [(acq_values[ind], configs[ind]) for ind in indices[::-1]]
    
    def _acquisition_function(self, configs, **kwargs):
        X = convert_configurations_to_array(configs)
        return self.acq(X, **kwargs)
    
    
    def _filter_excluded_configs(self, configs: List[Configuration], excluded_configs: List[Configuration]) -> List[Configuration]:
        """Filter out excluded configurations from candidates
        
        Parameters
        ----------
        configs : List[Configuration]
            Candidate configurations
        excluded_configs : List[Configuration]
            Configurations to exclude
            
        Returns
        -------
        List[Configuration]
            Filtered configurations
        """
        if not excluded_configs:
            return configs
        
        excluded_set = set()
        for config in excluded_configs:
            excluded_set.add(tuple(sorted(config.get_dictionary().items())))
        
        filtered = []
        for config in configs:
            config_key = tuple(sorted(config.get_dictionary().items()))
            if config_key not in excluded_set:
                filtered.append(config)
        
        return filtered
    
    def _prepare_observations_for_strategy(self, history:History, strategy, **kwargs) -> List[Any]:
        """Prepare observations for strategy by sorting by actual y value (standard BO approach)
        
        For LocalSearchGenerator, sort observations by actual y value (ascending, assuming minimization).
        
        Parameters
        ----------
        observations : List[Observation]
            Historical observations, each should have .y attribute
        strategy : SearchGenerator
            The strategy that will use these observations
        **kwargs
            Additional arguments (unused, kept for API compatibility)
            
        Returns
        -------
        List[Observation]
            Observations sorted by y value (ascending, best first), or original if sorting not needed
        """
        if isinstance(strategy, LocalSearchGenerator) and history:
            observations=history.observations
            sorted_observations = sorted(observations, key=lambda obs: obs.objectives[0])
            return sorted_observations
        return history.observations
    
    def reset(self):
        self.iter_id = 0


class CompositeOptimizer(AcquisitionOptimizer):
    """Composite Optimizer, use different search strategies [across] different iterations.
    
    Each iteration, the optimizer will select configurations sorted by acquisition value
    
    use the strategy pattern to combine multiple search strategies:
    1. use StrategySelector to select a strategy
    2. strategy generates candidate configurations
    3. batch evaluate all candidates' acquisition value
    4. select the best num_points configurations
    
    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        acquisition function
    config_space : ConfigurationSpace
        configuration space
    strategies : List[SearchGenerator]
        strategy list
    selector : StrategySelector
        strategy selector, if not provided, use FixedSelector(0)
    rng : np.random.RandomState
        random number generator, if not provided, use np.random.RandomState(42)
    candidate_multiplier : float, default=3.0
        candidate multiplier, generate num_points * candidate_multiplier candidates
        then select the best num_points configurations through acquisition function
        
    Examples
    --------
    >>> from .generator import LocalSearchGenerator, RandomSearchGenerator
    >>> from .selector import ProbabilisticSelector
    >>> 
    >>> # create strategy
    >>> local = LocalSearchGenerator(max_neighbors=50)
    >>> random = RandomSearchGenerator()
    >>> 
    >>> # create selector (85% local search, 15% random search)
    >>> selector = ProbabilisticSelector([0.85, 0.15])
    >>> 
    >>> # create composite optimizer
    >>> optimizer = CompositeOptimizer(
    ...     acquisition_function=acq_func,
    ...     config_space=config_space,
    ...     strategies=[local, random],
    ...     selector=selector
    ... )
    >>> 
    >>> # use
    >>> best_configs = optimizer.maximize(runhistory, num_points=10)
    """
    
    def __init__(self,
                 acquisition_function: AbstractAcquisitionFunction,
                 config_space: ConfigurationSpace,
                 strategies: List[SearchGenerator],
                 selector: StrategySelector = FixedSelector(0),
                 rng: np.random.RandomState = np.random.RandomState(42),
                 candidate_multiplier: float = 3.0):
        super().__init__(acquisition_function, config_space, rng)
        
        if not strategies:
            raise ValueError("At least one strategy is required")
        
        self.strategies = strategies
        self.selector = selector
        self.candidate_multiplier = candidate_multiplier
    
    def _maximize(self, history:History, num_points: int, excluded_configs: List[Configuration] = [], **kwargs) -> List[Tuple]:
        """use strategy to generate candidates, then batch evaluate and select the best num_points configurations
        
        process:
        1. use selector to select a strategy
        2. strategy generates candidates (generate num_points * candidate_multiplier candidates)
        3. batch evaluate all candidates' acquisition value
        4. select the best num_points configurations
        
        Parameters
        ----------
        observations : List[Any]
            historical observations
        num_points : int
            number of configurations to return
        excluded_configs : List[Configuration]
            configurations to exclude from generation
        **kwargs
            additional arguments passed to acquisition function
            
        Returns
        -------
        List[Tuple[float, Configuration]]
            list of (acquisition_value, configuration) pairs
        """
        strategy = self.selector.select(self.strategies, self.iter_id)
        logger.info(f"CompositeOptimizer: select strategy: {type(strategy).__name__}")
        
        ##sorted_observations = self._prepare_observations_for_strategy(history, strategy, **kwargs)
        n_candidates = int(num_points * self.candidate_multiplier)
        candidates = strategy.generate(
            history=history,
            num_points=n_candidates,
            rng=self.rng,
            acq_function=self.acq,
            **kwargs
        )
        
        if not candidates:
            raise RuntimeError(
                f"Strategy {type(strategy).__name__} generated no candidates. "
                "This should not happen if sampling_strategy is properly configured."
            )
        
        candidates = self._filter_excluded_configs(candidates, excluded_configs)
        if not candidates:
            raise RuntimeError("All generated candidates were excluded. Consider increasing candidate_multiplier.")

        scores = self._evaluate_batch(candidates, **kwargs)
        sorted_indices = np.argsort(scores)[::-1][: num_points]
        results = [(scores[idx], candidates[idx]) for idx in sorted_indices]
        self.iter_id += 1
        
        return results
    
    def reset(self):
        super().reset()
        if hasattr(self.selector, 'reset'):
            self.selector.reset()


class QuotaCompositeOptimizer(AcquisitionOptimizer):
    """Quota Composite Optimizer - ensure final configurations returned contain configurations from different strategies according to quotas
    
    The difference between CompositeOptimizer:
    - CompositeOptimizer: use different strategies across iterations, in each iteration, use the same strategy
    - QuotaCompositeOptimizer: each strategy independently sort and select top-k according to quotas, ensure diversity
    
    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        acquisition function
    config_space : ConfigurationSpace
        configuration space
    strategies : List[SearchGenerator]
        strategy list
    quotas : List[int]
        each strategy's quota, representing the number of points contributed by each strategy in the final return
        for example, [3, 1] means in each 4 points, 3 points come from strategy 0, 1 point come from strategy 1
    rng : np.random.RandomState
        random number generator
    candidate_multiplier : float, default=3.0
        candidate multiplier, each strategy generates quota * candidate_multiplier candidates
        
    Examples
    --------
    >>> local = LocalSearchGenerator(...)
    >>> random = RandomSearchGenerator(...)
    >>> 
    >>> # quotas=[3, 1] means in each 4 points, 3 points come from local, 1 point come from random
    >>> optimizer = QuotaCompositeOptimizer(
    ...     acquisition_function=acq_func,
    ...     config_space=config_space,
    ...     strategies=[local, random],
    ...     quotas=[3, 1]
    ... )
    >>> 
    >>> # return 8 points: 6 points from local + 2 points from random, interleaved
    >>> configs = optimizer.maximize(observations, num_points=8)
    """
    
    def __init__(self,
                 acquisition_function: AbstractAcquisitionFunction,
                 config_space: ConfigurationSpace,
                 strategies: List[SearchGenerator],
                 quotas: List[int],
                 rng: np.random.RandomState = np.random.RandomState(42),
                 candidate_multiplier: float = 3.0):
        super().__init__(acquisition_function, config_space, rng)
        
        if not strategies:
            raise ValueError("At least one strategy is required")
        if len(strategies) != len(quotas):
            raise ValueError(f"Number of strategies ({len(strategies)}) must match number of quotas ({len(quotas)})")
        if not all(q > 0 for q in quotas):
            raise ValueError("All quotas must be positive integers")
        
        self.strategies = strategies
        self.quotas = quotas
        self.total_quota = sum(quotas)
        self.candidate_multiplier = candidate_multiplier
    
    def _maximize(self, 
                  history:History, 
                  num_points: int, 
                  excluded_configs: List[Configuration] = [],
                  **kwargs) -> List[Tuple]:
        strategy_num_points = []
        remaining = num_points
        for i, quota in enumerate(self.quotas):
            if i == len(self.quotas) - 1:
                n = remaining
            else:
                n = int(np.ceil(num_points * quota / self.total_quota))
                n = min(n, remaining)
            strategy_num_points.append(n)
            remaining -= n
        
        strategy_results = []  # List[List[(score, config)]]
        for i, (strategy, n_points) in enumerate(zip(self.strategies, strategy_num_points)):
            if n_points <= 0:
                strategy_results.append([])
                continue
                
            logger.info(f"QuotaCompositeOptimizer: strategy {type(strategy).__name__} generating {n_points} points")
            
            ##sorted_observations = self._prepare_observations_for_strategy(history, strategy, **kwargs)
            
            n_candidates = int(n_points * self.candidate_multiplier)
            candidates = strategy.generate(
                history=history,
                num_points=n_candidates,
                rng=self.rng,
                acq_function=self.acq,
                **kwargs
            )
            
            if not candidates:
                logger.warning(f"Strategy {type(strategy).__name__} generated no candidates")
                strategy_results.append([])
                continue
            
            candidates = self._filter_excluded_configs(candidates, excluded_configs)
            if not candidates:
                logger.warning(f"Strategy {type(strategy).__name__}: all candidates were excluded")
                strategy_results.append([])
                continue
            
            scores = self._evaluate_batch(candidates, **kwargs)
            sorted_indices = np.argsort(scores)[::-1][: n_points]
            results = [(scores[idx], candidates[idx]) for idx in sorted_indices]
            logger.info(f"QuotaCompositeOptimizer: strategy {type(strategy).__name__} generated {len(results)} points, sorted by acquisition value: {scores[sorted_indices]}")
            strategy_results.append(results)
        
        final_results = self._interleave_results(strategy_results)
        self.iter_id += 1
        
        return final_results[: num_points]
    
    def _interleave_results(self, strategy_results: List[List[Tuple]]) -> List[Tuple]:
        """interleave the results from different strategies according to quotas
        
        for example, quotas=[3,1], strategy_results=[[a1,a2,a3,a4], [b1,b2]]
        final result: [a1, a2, a3, b1, a4, b2, ...]
        """
        result = []
        indices = [0] * len(self.strategies)
        
        while True:
            added_this_round = False
            
            for strategy_idx, quota in enumerate(self.quotas):
                results = strategy_results[strategy_idx]
                idx = indices[strategy_idx]
                
                for _ in range(quota):
                    if idx < len(results):
                        result.append(results[idx])
                        idx += 1
                        added_this_round = True
                
                indices[strategy_idx] = idx
            
            if not added_this_round:
                break
        
        return result
