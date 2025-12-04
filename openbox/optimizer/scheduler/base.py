# License: MIT
# Author: LINGCHING TUNG
import abc
import numpy as np
from typing import List, Tuple

class BaseScheduler(abc.ABC):
    def __init__(self, num_nodes: int = 1):
        self.num_nodes = num_nodes
        self.fidelity_levels = [round(float(1.0), 5)]

    def get_bracket_index(self, iter_id: int) -> int:
        # always return 0 when using full fidelity scheduler since there is only one fidelity level
        return iter_id % len(self.fidelity_levels)

    @abc.abstractmethod
    def get_elimination_count(self) -> int:
        pass

    def eliminate_candidates(
        self, candidates: List, perfs: List, **kwargs
    ) -> Tuple[List, List]:
        reduced_num = self.get_elimination_count(**kwargs)
        indices = np.argsort(perfs)
        sorted_candidates = [candidates[i] for i in indices]
        sorted_perfs = [perfs[i] for i in indices]
        return sorted_candidates[:reduced_num], sorted_perfs[:reduced_num]

    def get_fidelity_levels(self) -> List[float]:
        return self.fidelity_levels

    @abc.abstractmethod
    def calculate_resource_ratio(self) -> float:
        pass

    def get_stage_params(self, **kwargs) -> Tuple[int, int]:
        return self.num_nodes, 1

    def should_update_history(self, resource_ratio: float) -> bool:
        """
        Determine whether observations should be updated to advisor.history.
        Notes:
            - For SMBO (FullFidelityScheduler): always returns True
            - For BOHB: only returns True when resource_ratio == 1.0
            - For MFSE: returns True
        """
        return True


class FullFidelityScheduler(BaseScheduler):
    def __init__(self, num_nodes: int = 1, **kwargs):
        super().__init__(num_nodes)

    def calculate_resource_ratio(self, **kwargs) -> float:
        return round(float(1.0), 5)

    def get_elimination_count(self, **kwargs) -> int:
        return self.num_nodes