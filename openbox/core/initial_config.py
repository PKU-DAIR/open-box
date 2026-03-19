# License: MIT
# Author: LINGCHING TUNG

"""
Unified Initial Configuration Provider.

Combines multiple sources of initial configurations:
1. User-provided configurations (highest priority)
2. Warm-start from transfer learning history
3. Initial design strategies (sobol, latin_hypercube, etc.)
4. Random sampling (fallback)
"""

from typing import List, Optional, Dict, Tuple
from ConfigSpace import Configuration, ConfigurationSpace
import numpy as np

from openbox import logger
from openbox.utils.history import History
from openbox.utils.samplers import SobolSampler, LatinHypercubeSampler, HaltonSampler


class InitialConfigProvider:
    """
    Unified initial configuration provider.
    
    Combines multiple sources of initial configurations with priority:
    1. User-provided configurations (highest priority)
    2. Warm-start from transfer learning history
    3. Initial design strategies (sobol, latin_hypercube, etc.)
    4. Random sampling (fallback)
    
    Parameters
    ----------
    config_space : ConfigurationSpace
        Configuration space.
    init_num : int, default=3
        Total number of initial configurations to use.
    init_strategy : str, default='random_explore_first'
        Strategy for generating initial design configs:
        - 'random': Pure random sampling
        - 'default': Default config + random sampling
        - 'random_explore_first': Max-min distance (space-filling)
        - 'sobol': Sobol sequence sampling
        - 'latin_hypercube': Latin hypercube sampling
        - 'halton': Halton sequence sampling
    initial_configurations : List[Configuration], optional
        User-provided initial configurations (highest priority).
    transfer_learning_history : List[History], optional
        Historical data for warm start.
    warm_start_strategy : str, default='topk'
        How to select configs from transfer learning history:
        - 'no': Do not use warm start even if transfer_learning_history is provided
        - 'best': Select best config from each source
        - 'topk': Select top-k configs from each history
    warm_start_num : int, optional
        Number of configs to extract from transfer learning (independent of init_num).
        If None and has transfer learning history, uses init_num.
        Note: warm_start configs may not all be used for init - some can be used 
        in multi-fidelity competition pool.
    rng : RandomState, optional
        Random number generator.
        
    Attributes
    ----------
    init_num : int
        Number of initial configurations.
    config_queue : List[Configuration]
        Queue of initial configurations.
    config_sources : List[str]
        Source of each configuration in the queue.
        
    Examples
    --------
    >>> from openbox.core.initial_config import InitialConfigProvider
    >>> provider = InitialConfigProvider(
    ...     config_space=cs,
    ...     init_num=5,
    ...     init_strategy='sobol',
    ...     transfer_learning_history=histories
    ... )
    >>> for i in range(len(provider)):
    ...     config = provider.get_config(i)
    ...     source = provider.get_config_source(i)
    ...     print(f"Config {i} from {source}")
    """
    
    def __init__(
        self,
        config_space: ConfigurationSpace,
        init_num: int = 3,
        init_strategy: str = 'random_explore_first',
        initial_configurations: Optional[List[Configuration]] = None,
        transfer_learning_history: Optional[List[History]] = None,
        warm_start_strategy: str = 'best',
        warm_start_num: Optional[int] = None,
        rng: np.random.RandomState = None,
    ):
        self.config_space = config_space
        self._init_num = init_num
        self.init_strategy = init_strategy
        self.warm_start_strategy = warm_start_strategy
        self.rng = rng if rng is not None else np.random.RandomState()
        
        # Configuration queue and sources
        self._config_queue: List[Configuration] = []
        self._config_sources: List[str] = []
        
        # Build the configuration queue
        self._build_config_queue(
            initial_configurations=initial_configurations,
            transfer_learning_history=transfer_learning_history,
            warm_start_num=warm_start_num
        )
        
        logger.info(f'InitialConfigProvider initialized: {len(self._config_queue)} configs ready. '
                   f'Sources: {self.get_source_summary()}')
    
    def _build_config_queue(
        self,
        initial_configurations: Optional[List[Configuration]],
        transfer_learning_history: Optional[List[History]],
        warm_start_num: Optional[int]
    ):        
        # 1. User-provided configurations (highest priority)
        if initial_configurations is not None and len(initial_configurations) > 0:
            for config in initial_configurations:
                self._add_config(config, source='user_provided')
            logger.info(f'Added {len(initial_configurations)} user-provided configurations')
        
        # 2. Warm-start from transfer learning history (warm_start_num is independent of init_num)
        if (transfer_learning_history is not None 
            and len(transfer_learning_history) > 0 
            and self.warm_start_strategy != 'no'):
            num_warm = warm_start_num if warm_start_num is not None else self._init_num
            warm_configs = self._extract_warm_start_configs(
                histories=transfer_learning_history,
                strategy=self.warm_start_strategy,
                num=num_warm
            )
            for config in warm_configs:
                self._add_config(config, source='warm_start')
            
            logger.info(f'Warm start: extracted {len(warm_configs)} configs (warm_start_num={num_warm})')
        
        # 3. Initial design configurations (fill remaining to reach init_num)
        remaining = self._init_num - len(self._config_queue)
        if remaining > 0:
            design_configs = self._create_initial_design(remaining)
            for config in design_configs:
                self._add_config(config, source=f'initial_design:{self.init_strategy}')
        
        # 4. Random fallback (if still not enough)
        remaining = self._init_num - len(self._config_queue)
        if remaining > 0:
            logger.warning(f'Still need {remaining} configs, filling with random sampling')
            random_configs = self._sample_random(
                remaining, 
                excluded=set(self._config_queue)
            )
            for config in random_configs:
                self._add_config(config, source='random_fallback')
    
    def _add_config(self, config: Configuration, source: str) -> bool:
        """
        Add config to queue if valid and not duplicate.
        
        Returns True if added successfully, False otherwise.
        """
        if config in self._config_queue:
            logger.debug(f'Skipping duplicate config from {source}')
            return False
        config.origin = source
        self._config_queue.append(config)
        self._config_sources.append(source)
        return True
    
    def _extract_warm_start_configs(
        self, 
        histories: List[History],
        strategy: str,
        num: int
    ) -> List[Configuration]:
        """
        Extract configurations from transfer learning history.
        
        Parameters
        ----------
        histories : List[History]
            Historical data from source tasks.
        strategy : str
            Extraction strategy ('best', 'topk').
        num : int
            Number of configurations to extract.
            
        Returns
        -------
        configs : List[Configuration]
            Extracted configurations.
        """
        configs = []
        
        if strategy == 'best':
            for hist in histories:
                if len(hist) > 0:
                    try:
                        incumbent_config = hist.get_incumbents()[0]
                        if incumbent_config is not None and incumbent_config not in configs:
                            configs.append(incumbent_config)
                            if len(configs) >= num:
                                break
                    except Exception as e:
                        logger.debug(f'Failed to get incumbent from history: {e}')
                        continue
        
        elif strategy == 'topk':
            # Select top-k configs from each history until reaching num
            # k_per_hist: how many to take from each history per round
            k_per_hist = max(1, (num + len(histories) - 1) // len(histories)) if len(histories) > 0 else num
            
            # Round-robin selection: take top-k from each history until we have enough
            # This ensures balanced selection across all histories
            hist_iterators = []
            for hist in histories:
                if len(hist) == 0:
                    continue
        
                hist_configs_with_perf: List[Tuple[Configuration, float]] = []
                for obs in hist.observations:
                    if obs.objectives is not None and len(obs.objectives) > 0:
                        hist_configs_with_perf.append((obs.config, obs.objectives[0]))
                
                hist_configs_with_perf.sort(key=lambda x: x[1])
                hist_iterators.append(iter(hist_configs_with_perf))
            
            # Take top-k from each history in round-robin fashion
            while len(configs) < num and hist_iterators:
                exhausted = []
                for i, it in enumerate(hist_iterators):
                    count = 0
                    while count < k_per_hist and len(configs) < num:
                        try:
                            config, _ = next(it)
                            if config not in configs:
                                configs.append(config)
                                count += 1
                        except StopIteration:
                            exhausted.append(i)
                            break

                for i in reversed(exhausted):
                    hist_iterators.pop(i)
            
            remaining = num - len(configs)
            if remaining > 0:
                all_configs = []
                for hist in histories:
                    all_configs.extend([obs.config for obs in hist.observations])
                
                self.rng.shuffle(all_configs)
                for config in all_configs:
                    if config not in configs:
                        configs.append(config)
                        if len(configs) >= num:
                            break        
        else:
            raise ValueError(f'Unknown warm start strategy: {strategy}. '
                           f'Supported: best, topk')
        
        logger.info(f'Warm start: extracted {len(configs)} configs using "{strategy}" strategy')
        return configs
    
    def _create_initial_design(self, num: int) -> List[Configuration]:
        """
        Generate initial design configurations.
        
        Parameters
        ----------
        num : int
            Number of configurations to generate.
            
        Returns
        -------
        configs : List[Configuration]
            Generated configurations.
        """
        if num <= 0:
            return []
        
        default_config = self.config_space.get_default_configuration()
        
        num_random = max(0, num - 1)
        configs = []
        
        if self.init_strategy == 'random':
            configs = self._sample_random(num, excluded=set(self._config_queue))
            
        elif self.init_strategy == 'default':
            if default_config not in self._config_queue:
                configs = [default_config]
            configs.extend(self._sample_random(
                num - len(configs), 
                excluded=set(self._config_queue + configs)
            ))
            
        elif self.init_strategy == 'random_explore_first':
            candidates = self._sample_random(max(100, num * 10), excluded=set(self._config_queue))
            configs = self._max_min_distance(
                default_config, 
                candidates, 
                num_random
            )
            
        elif self.init_strategy == 'sobol':
            try:
                sampler = SobolSampler(self.config_space, num_random, random_state=self.rng)
                sobol_configs = sampler.generate(return_config=True)
                if default_config not in self._config_queue:
                    configs = [default_config] + sobol_configs
                else:
                    configs = sobol_configs
            except Exception as e:
                logger.warning(f'Sobol sampling failed: {e}. Falling back to random.')
                configs = self._sample_random(num, excluded=set(self._config_queue))
                
        elif self.init_strategy == 'latin_hypercube':
            try:
                sampler = LatinHypercubeSampler(self.config_space, num_random, criterion='maximin')
                lhs_configs = sampler.generate(return_config=True)
                if default_config not in self._config_queue:
                    configs = [default_config] + lhs_configs
                else:
                    configs = lhs_configs
            except Exception as e:
                logger.warning(f'Latin hypercube sampling failed: {e}. Falling back to random.')
                configs = self._sample_random(num, excluded=set(self._config_queue))
                
        elif self.init_strategy == 'halton':
            try:
                sampler = HaltonSampler(self.config_space, num_random, random_state=self.rng)
                halton_configs = sampler.generate(return_config=True)
                if default_config not in self._config_queue:
                    configs = [default_config] + halton_configs
                else:
                    configs = halton_configs
            except Exception as e:
                logger.warning(f'Halton sampling failed: {e}. Falling back to random.')
                configs = self._sample_random(num, excluded=set(self._config_queue))
        else:
            raise ValueError(f'Unknown initial design strategy: {self.init_strategy}. '
                           f'Supported: random, default, random_explore_first, sobol, latin_hypercube, halton')
        
        # Validate and filter
        valid_configs = []
        for config in configs:
            if config in self._config_queue or config in valid_configs:
                continue
            valid_configs.append(config)

        return valid_configs[: num]
    
    def _sample_random(
        self, 
        num: int, 
        excluded: Optional[set] = None
    ) -> List[Configuration]:
        if num <= 0:
            return []
        
        if excluded is None:
            excluded = set()
        
        configs = []
        max_trials = max(1000, num * 20)
        trials = 0
        
        while len(configs) < num and trials < max_trials:
            trials += 1
            config = self.config_space.sample_configuration()
            if config not in configs and config not in excluded:
                configs.append(config)
        
        if len(configs) < num:
            logger.warning(f'Could only sample {len(configs)}/{num} random configs after {max_trials} trials')
        
        return configs
    
    def _max_min_distance(
        self, 
        default_config: Configuration,
        candidates: List[Configuration], 
        num: int,
    ) -> List[Configuration]:
        """
        Select configurations maximizing minimum distance (space-filling).
        
        Parameters
        ----------
        default_config : Configuration
            Default configuration to start with.
        candidates : List[Configuration]
            Candidate configurations.
        num : int
            Number of configurations to select (excluding default_config).
            
        Returns
        -------
        configs : List[Configuration]
            Selected configurations (including default_config).
        """
        initial_configs = [default_config]
        
        if len(candidates) == 0:
            return initial_configs
        
        min_dis = np.array([
            np.linalg.norm(config.get_array() - default_config.get_array())
            for config in candidates
        ])
        
        num_to_select = min(num, len(candidates))
        
        for _ in range(num_to_select):
            idx = np.argmax(min_dis)
            if min_dis[idx] <= 0:
                break
            
            furthest_config = candidates[idx]
            initial_configs.append(furthest_config)
            min_dis[idx] = -1

            for j in range(len(candidates)):
                if min_dis[j] > 0:  # Only update unselected configs
                    updated_dis = np.linalg.norm(candidates[j].get_array() - furthest_config.get_array())
                    min_dis[j] = min(updated_dis, min_dis[j])

        return initial_configs
    
    # ========== Public API ==========
    
    def get_config(self, index: int) -> Optional[Configuration]:
        """
        Get initial configuration by index.
        
        Parameters
        ----------
        index : int
            Index of the configuration.
            
        Returns
        -------
        config : Configuration or None
            Configuration at the index, or None if exhausted.
        """
        if 0 <= index < len(self._config_queue):
            return self._config_queue[index]
        return None
    
    def get_config_source(self, index: int) -> Optional[str]:
        """
        Get the source of configuration at index.
        
        Parameters
        ----------
        index : int
            Index of the configuration.
            
        Returns
        -------
        source : str or None
            Source string, or None if index out of range.
        """
        if 0 <= index < len(self._config_sources):
            return self._config_sources[index]
        return None
    
    def is_exhausted(self, num_evaluated: int) -> bool:
        """
        Check if all initial configurations have been used.
        
        Parameters
        ----------
        num_evaluated : int
            Number of configurations already evaluated.
            
        Returns
        -------
        exhausted : bool
            True if all initial configs have been used.
        """
        return num_evaluated >= len(self._config_queue)
    
    def get_source_summary(self) -> Dict[str, int]:
        """
        Get summary of configuration sources.
        
        Returns
        -------
        summary : dict
            Dictionary mapping source type to count.
        """
        summary = {}
        for source in self._config_sources:
            # Extract main source type (before ':')
            key = source.split(':')[0]
            summary[key] = summary.get(key, 0) + 1
        return summary
    
    @property
    def config_queue(self) -> List[Configuration]:
        """List of initial configurations (read-only copy)."""
        return list(self._config_queue)
    
    @property
    def config_sources(self) -> List[str]:
        """List of configuration sources (read-only copy)."""
        return list(self._config_sources)
    
    def __len__(self) -> int:
        """Number of initial configurations."""
        return len(self._config_queue)
    
    def __repr__(self) -> str:
        return (f"InitialConfigProvider(init_num={len(self)}, "
                f"strategy='{self.init_strategy}', "
                f"sources={self.get_source_summary()})")
