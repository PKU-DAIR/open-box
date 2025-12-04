# License: MIT
# Author: LINGCHING TUNG
import numpy as np
from math import log, ceil
from typing import List, Tuple
from openbox import logger

from .base import BaseScheduler

class FixedFidelityScheduler(BaseScheduler):
    def __init__(self, 
                 n_resources: List[int],
                 r_resources: List[int],
                 fidelity_levels: List[float],
                 num_nodes: int = 1):
        super().__init__(num_nodes)
        self.n_resources = n_resources
        self.r_resources = r_resources
        self.fidelity_levels = fidelity_levels

        for r in self.r_resources:
            if r not in self.fidelity_levels:
                raise ValueError(f"r_resource {r} not in fidelity_levels {self.fidelity_levels}")

    def get_stage_params(self, stage: int, **kwargs) -> Tuple[int, int]:
        assert stage < len(self.n_resources) and stage < len(self.r_resources), "Stage index out of range"
        return self.n_resources[stage] * self.num_nodes, self.r_resources[stage]

    def get_elimination_count(self, stage: int, **kwargs) -> int:
        reduced_num = self.n_resources[stage + 1] if stage + 1 < len(self.n_resources) else self.n_resources[-1]
        return reduced_num * self.num_nodes

    def calculate_resource_ratio(self, n_resource: int) -> float:
        return round(float(n_resource), 5)
    
    def should_update_history(self, resource_ratio: float) -> bool:
        return True

class BOHBFidelityScheduler(BaseScheduler):
    """
    Multi-Fidelity Scheduler for BOHB optimization.
    
    This class provides the core scheduling logic:
    - Determines how many configurations to run (n)
    - Determines how much resource to allocate (r)
    - Manages bracket and stage structure
    - Calculates elimination counts
    """
    
    def __init__(self, 
                 num_nodes: int = 1,
                 R: int = 9, eta: int = 3):
        super().__init__(num_nodes)
        self.R = R
        self.eta = eta        
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.s_values = list(reversed(range(self.s_max + 1)))

        self.fidelity_levels = [round(x / self.R, 5) for x in np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)]
        assert len(self.fidelity_levels) == self.s_max + 1, "Fidelity levels length mismatch"
    
        logger.info("FidelityScheduler: run %d brackets with fidelity levels %s. s_max = [%d]. R = [%d], eta = [%d]" 
                    % (len(self.s_values), self.get_fidelity_levels(), self.s_max, self.R, self.eta))
    
    def get_bracket_index(self, iter_id: int) -> int:
        return self.s_values[iter_id % len(self.s_values)]

    def get_bracket_params(self, s: int) -> Tuple[int, int]:
        """
        Get bracket parameters for a given bracket index.
        
        Args:
            s: Bracket index (0 to s_max)
            
        Returns:
            Tuple of (n_configs, n_resource)
        """
        n_configs = int(ceil(self.B / self.R / (s + 1) * self.eta ** s)) * self.num_nodes
        n_resource = int(self.R * self.eta ** (-s))
        return n_configs, n_resource
    
    def get_stage_params(self, s: int, stage: int) -> Tuple[int, int]:
        """
        Get stage parameters within a bracket.
        
        Args:
            s: Bracket index
            stage: Stage index within bracket (0 to s)
            num_nodes: Number of Spark nodes (for multi-node scaling)
            
        Returns:
            Tuple of (n_configs, n_resource)
        """
        n_configs, base_resource = self.get_bracket_params(s)
        n_configs_stage = int(n_configs * self.eta ** (-stage))
        n_resource_stage = int(base_resource * self.eta ** stage)
        
        return n_configs_stage, n_resource_stage
    
    def calculate_resource_ratio(self, n_resource: int) -> float:
        """
        Calculate resource ratio from resource allocation.
        
        Args:
            n_resource: Resource allocation
            
        Returns:
            Resource ratio (0.0 to 1.0)
        """
        return round(float(n_resource / self.R), 5)
    
    def get_elimination_count(self, s: int, stage: int) -> int:
        """
        Get number of configurations to eliminate after a stage.
        
        Args:
            s: Bracket index
            stage: Stage index
            
        Returns:
            Number of configurations to keep
        """
        n_configs, r_resource = self.get_stage_params(s, stage)
        return int(n_configs / self.eta) if int(r_resource) != self.R else int(n_configs)
    
    def should_update_history(self, resource_ratio: float) -> bool:
        # only update history when resource_ratio == 1.0
        return resource_ratio == round(float(1.0), 5)


class MFESFidelityScheduler(BOHBFidelityScheduler):
    def __init__(self, 
                 num_nodes: int = 1,
                 R: int = 9, eta: int = 3):
        super().__init__(num_nodes=num_nodes, R=R, eta=eta)

    def should_update_history(self, resource_ratio: float) -> bool:
        # always return True for MFSE - let MFBO.update decide history vs history_list
        return True


class FlattenFidelityScheduler(BOHBFidelityScheduler):
    """
    Scheduler with expanded full-fidelity brackets.
    
    This scheduler is similar to BOHBFidelityScheduler, but expands the last 
    full-fidelity bracket (s=0) into multiple single-configuration brackets.
    
    For example, if the last bracket would be (r=27, n=4), it creates 4 separate 
    brackets each with (r=27, n=1). This allows more fine-grained scheduling of 
    full-fidelity evaluations.
    
    The bracket structure stores explicit (n_configs, n_resource) tuples instead
    of using s indices.
    """
    
    def __init__(self, 
                 num_nodes: int = 1,
                 R: int = 9, eta: int = 3):
        super().__init__(num_nodes=num_nodes, R=R, eta=eta)
        
        self.brackets = []
        
        for s in range(self.s_max, 0, -1):
            n_configs = int(ceil(self.B / self.R / (s + 1) * self.eta ** s)) * self.num_nodes
            n_resource = int(self.R * self.eta ** (-s))
            
            stages = []
            for stage in range(s + 1):
                n_configs_stage = int(n_configs * self.eta ** (-stage))
                n_resource_stage = int(n_resource * self.eta ** stage)
                stages.append((n_configs_stage, n_resource_stage))
            
            self.brackets.append({
                's': s,
                'n_configs': n_configs,
                'n_resource': n_resource,
                'stages': stages
            })
        
        # Expand the last bracket (s=0) into multiple single-config brackets
        s = 0
        n_configs_last = int(ceil(self.B / self.R / (s + 1) * self.eta ** s)) * self.num_nodes
        n_resource_last = int(self.R * self.eta ** (-s))
        num_expanded = n_configs_last // self.num_nodes
        
        for i in range(num_expanded):
            self.brackets.append({
                's': 0,
                'expanded_idx': i,
                'n_configs': self.num_nodes,
                'n_resource': n_resource_last,
                'stages': [(self.num_nodes, n_resource_last)]
            })
        
        logger.info(f"FlattenFidelityScheduler: Expanded last bracket (r={n_resource_last}, n={n_configs_last}) "
                    f"into {num_expanded} brackets of (r={n_resource_last}, n={self.num_nodes})")
        logger.info(f"Total brackets: {len(self.brackets)}")
        
    def get_bracket_index(self, iter_id: int) -> int:
        bracket_idx = iter_id % len(self.brackets)
        bracket = self.brackets[bracket_idx]
        self._current_bracket_idx = bracket_idx
        return bracket['s']
    
    def get_bracket_params(self, s: int) -> Tuple[int, int]:
        if hasattr(self, '_current_bracket_idx'):
            bracket = self.brackets[self._current_bracket_idx]
            if bracket['s'] == s:
                return bracket['n_configs'], bracket['n_resource']
        for bracket in self.brackets:
            if bracket['s'] == s:
                return bracket['n_configs'], bracket['n_resource']
        raise ValueError(f"Bracket with s={s} not found")
    
    def get_stage_params(self, s: int, stage: int) -> Tuple[int, int]:
        if hasattr(self, '_current_bracket_idx'):
            bracket = self.brackets[self._current_bracket_idx]
            if bracket['s'] == s and stage < len(bracket['stages']):
                return bracket['stages'][stage]
        for bracket in self.brackets:
            if bracket['s'] == s:
                if stage < len(bracket['stages']):
                    return bracket['stages'][stage]
                else:
                    raise ValueError(f"Stage {stage} out of range for bracket with s={s}")
        raise ValueError(f"Bracket with s={s} not found or stage {stage} out of range")
    
    def get_elimination_count(self, s: int, stage: int) -> int:
        if hasattr(self, '_current_bracket_idx'):
            bracket = self.brackets[self._current_bracket_idx]
            if bracket['s'] == s and stage < len(bracket['stages']):
                n_configs, r_resource = bracket['stages'][stage]
                if stage == len(bracket['stages']) - 1 or r_resource == self.R:
                    return n_configs
                else:
                    return int(n_configs / self.eta)
        for bracket in self.brackets:
            if bracket['s'] == s:
                if stage < len(bracket['stages']):
                    n_configs, r_resource = bracket['stages'][stage]
                    if stage == len(bracket['stages']) - 1 or r_resource == self.R:
                        return n_configs
                    else:
                        return int(n_configs / self.eta)
        raise ValueError(f"Bracket with s={s} not found or stage {stage} out of range")
    

class MFESFlattenFidelityScheduler(FlattenFidelityScheduler):
    def __init__(self, 
                 num_nodes: int = 1,
                 R: int = 9, eta: int = 3):
        super().__init__(num_nodes=num_nodes, R=R, eta=eta)

    def should_update_history(self, resource_ratio: float) -> bool:
        return True