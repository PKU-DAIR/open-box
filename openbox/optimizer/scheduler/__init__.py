# License: MIT
# Author: LINGCHING TUNG
from .base import BaseScheduler, FullFidelityScheduler
from .fidelity import FixedFidelityScheduler, \
    BOHBFidelityScheduler, MFESFidelityScheduler, \
    FlattenFidelityScheduler, MFESFlattenFidelityScheduler

schedulers = {
    'fixed': FixedFidelityScheduler,
    'bohb': BOHBFidelityScheduler,
    'full': FullFidelityScheduler,
    'mfes': MFESFidelityScheduler,
    'flatten': FlattenFidelityScheduler,
    'bohb_flatten': FlattenFidelityScheduler,
    'mfes_flatten': MFESFlattenFidelityScheduler
}

__all__ = [
    'BaseScheduler',
    'FullFidelityScheduler',
    'FixedFidelityScheduler',
    'BOHBFidelityScheduler',
    'MFESFidelityScheduler',
    'FlattenFidelityScheduler',
    'MFESFlattenFidelityScheduler',
    'schedulers'
]