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


def build_scheduler(scheduler_type: str = 'full', **kwargs) -> BaseScheduler:
    """
    Factory function to create scheduler instance.
    
    Args:
        scheduler_type: Type of scheduler
            - 'full': Full fidelity scheduler (default)
            - 'bohb': BOHB-style successive halving
            - 'mfes': MFES-style multi-fidelity
            - 'flatten': Flattened BOHB scheduler
            - 'bohb_flatten': Alias for flatten
            - 'mfes_flatten': Flattened MFES scheduler
            - 'fixed': Fixed fidelity levels
        **kwargs: Additional arguments passed to scheduler constructor
        
    Returns:
        Scheduler instance
        
    Raises:
        ValueError: If scheduler_type is unknown
    """
    if scheduler_type not in schedulers:
        raise ValueError(
            f'Unknown scheduler_type: {scheduler_type}. '
            f'Available options: {list(schedulers.keys())}'
        )
    
    scheduler_class = schedulers[scheduler_type]
    return scheduler_class(**kwargs)


def check_scheduler(objective_function: callable, scheduler_type: str = 'full') -> bool:
    """
    Check if the objective function supports the scheduler type.
    
    Args:
        objective_function: Objective function
        scheduler_type: Type of scheduler
    Returns:
        True if the objective function supports the scheduler type, False otherwise
    """
    import inspect
    sig = inspect.signature(objective_function)
    if scheduler_type != 'full' and 'resource_ratio' not in sig.parameters:
        raise ValueError(
            f'Multi-fidelity scheduler "{scheduler_type}" requires objective function '
            f'to accept "resource_ratio" parameter.\n'
            f'Please modify your objective function signature to:\n'
            f'  def objective_function(config, resource_ratio=1.0):\n'
            f'      ...\n'
            f'Or use scheduler_type="full" for single-fidelity optimization.'
        )
    return True


__all__ = [
    'BaseScheduler',
    'FullFidelityScheduler',
    'FixedFidelityScheduler',
    'BOHBFidelityScheduler',
    'MFESFidelityScheduler',
    'FlattenFidelityScheduler',
    'MFESFlattenFidelityScheduler',
    'schedulers',
    'build_scheduler',
    'check_scheduler'
]