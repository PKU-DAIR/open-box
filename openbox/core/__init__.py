# License: MIT
# Author: LINGCHING TUNG

from openbox import logger


# Advisor registry mapping advisor type to (class, required_conditions)
_ADVISOR_REGISTRY = {}


def register_advisor(name, advisor_class, required_conditions=None):
    """
    Register an advisor class with optional requirements.
    
    Parameters
    ----------
    name : str
        Advisor type name
    advisor_class : type
        Advisor class
    required_conditions : dict, optional
        Requirements like {'num_objectives': 1, 'num_constraints': 0}
    """
    _ADVISOR_REGISTRY[name] = {
        'class': advisor_class,
        'conditions': required_conditions or {}
    }


def _check_advisor_conditions(advisor_type, num_objectives, num_constraints):
    if advisor_type not in _ADVISOR_REGISTRY:
        return True
    
    conditions = _ADVISOR_REGISTRY[advisor_type]['conditions']
    
    if 'num_objectives' in conditions:
        if num_objectives != conditions['num_objectives']:
            raise ValueError(
                f"Advisor '{advisor_type}' requires num_objectives={conditions['num_objectives']}, "
                f"but got {num_objectives}"
            )
    
    if 'num_constraints' in conditions:
        if num_constraints != conditions['num_constraints']:
            raise ValueError(
                f"Advisor '{advisor_type}' requires num_constraints={conditions['num_constraints']}, "
                f"but got {num_constraints}"
            )
    
    return True


def build_advisor(
    advisor_type='default',
    config_space=None,
    num_objectives=1,
    num_constraints=0,
    initial_trials=3,
    init_strategy='random_explore_first',
    initial_configurations=None,
    optimization_strategy='bo',
    surrogate_type='auto',
    acq_type='auto',
    acq_optimizer_type='auto',
    ref_point=None,
    transfer_learning_history=None,
    early_stop=False,
    early_stop_kwargs=None,
    task_id='OpenBox',
    output_dir='logs',
    random_state=None,
    logger_kwargs=None,
    scheduler_type=None,
    # Batch advisor specific parameters
    batch_size=None,
    batch_strategy='default',
    **advisor_kwargs
):
    """
    Factory function to build advisor instances.
    
    Parameters
    ----------
    advisor_type : str, default='default'
        Type of advisor to create:
        - 'default': Generic Bayesian Optimization advisor
        - 'mf': Multi-fidelity Bayesian Optimization advisor
        - 'mcadvisor': Monte Carlo advisor
        - 'tpe': Tree-structured Parzen Estimator
        - 'ea': Evolutionary Algorithm advisor
        - 'random': Random search advisor
        - 'sync_batch': Synchronous batch advisor
        - 'async_batch': Asynchronous batch advisor
    config_space : ConfigSpace
        Configuration space
    num_objectives : int
        Number of objectives
    num_constraints : int
        Number of constraints
    initial_trials : int
        Number of initial trials
    init_strategy : str
        Initialization strategy
    initial_configurations : list, optional
        Initial configurations
    optimization_strategy : str
        Optimization strategy
        - 'bo' (default)
        - 'random'
        - 'ea'
    surrogate_type : str
        Surrogate model type
    acq_type : str
        Acquisition function type
    acq_optimizer_type : str
        Acquisition optimizer type
    ref_point : list, optional
        Reference point for multi-objective
    transfer_learning_history : list, optional
        Transfer learning history
    early_stop : bool
        Enable early stopping
    early_stop_kwargs : dict, optional
        Early stop parameters
    task_id : str
        Task identifier
    output_dir : str
        Output directory
    random_state : int, optional
        Random seed
    logger_kwargs : dict, optional
        Logger parameters
    batch_size : int, optional
        Batch size for batch advisors
    batch_strategy : str, optional
        Batch strategy for batch advisors
    **advisor_kwargs
        Additional advisor-specific parameters
        
    Returns
    -------
    advisor : BaseAdvisor
        Configured advisor instance
        
    Raises
    ------
    ValueError
        If advisor_type is invalid or requirements not met
    """
    advisor_type = advisor_type.lower()
    scheduler_type = scheduler_type.lower() if isinstance(scheduler_type, str) else scheduler_type

    mf_scheduler_types = {'mfes', 'mfes_flatten'}
    bo_scheduler_types = {'full', 'fixed', 'bohb', 'flatten', 'bohb_flatten'}
    if scheduler_type in mf_scheduler_types:
        if advisor_type != 'mf':
            logger.warning(
                'scheduler_type=%s requires mf advisor; override advisor_type from %s to mf.'
                % (scheduler_type, advisor_type)
            )
            advisor_type = 'mf'
        if surrogate_type == 'auto':
            surrogate_type = 'mfgpe'
    elif scheduler_type in bo_scheduler_types:
        if advisor_type == 'mf':
            logger.warning(
                'scheduler_type=%s should use regular BO advisor; override advisor_type from mf to default.'
                % scheduler_type
            )
            advisor_type = 'default'
        if surrogate_type == 'mfgpe':
            logger.warning(
                'scheduler_type=%s should not use mfgpe surrogate directly; override surrogate_type from mfgpe to auto.'
                % scheduler_type
            )
            surrogate_type = 'auto'

    if advisor_type == 'mf' and surrogate_type == 'auto':
        surrogate_type = 'mfgpe'
    
    _check_advisor_conditions(advisor_type, num_objectives, num_constraints)
    
    _logger_kwargs = logger_kwargs or {}
    if 'force_init' not in _logger_kwargs:
        _logger_kwargs['force_init'] = True
    
    if advisor_type == 'default':
        from openbox.core.generic_advisor import Advisor
        return Advisor(
            config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            initial_trials=initial_trials,
            init_strategy=init_strategy,
            initial_configurations=initial_configurations,
            optimization_strategy=optimization_strategy,
            surrogate_type=surrogate_type,
            acq_type=acq_type,
            acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point,
            transfer_learning_history=transfer_learning_history,
            early_stop=early_stop,
            early_stop_kwargs=early_stop_kwargs,
            task_id=task_id,
            output_dir=output_dir,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )

    elif advisor_type == 'mf':
        from openbox.core.mf_advisor import MFAdvisor
        return MFAdvisor(
            config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            initial_trials=initial_trials,
            init_strategy=init_strategy,
            initial_configurations=initial_configurations,
            optimization_strategy=optimization_strategy,
            surrogate_type=surrogate_type,
            acq_type=acq_type,
            acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point,
            transfer_learning_history=transfer_learning_history,
            early_stop=early_stop,
            early_stop_kwargs=early_stop_kwargs,
            task_id=task_id,
            output_dir=output_dir,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )
    
    elif advisor_type == 'mcadvisor':
        from openbox.core.mc_advisor import MCAdvisor
        return MCAdvisor(
            config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            initial_trials=initial_trials,
            init_strategy=init_strategy,
            initial_configurations=initial_configurations,
            optimization_strategy=optimization_strategy,
            surrogate_type=surrogate_type,
            acq_type=acq_type,
            acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point,
            transfer_learning_history=transfer_learning_history,
            task_id=task_id,
            output_dir=output_dir,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )
    
    elif advisor_type == 'tpe':
        from openbox.core.tpe_advisor import TPE_Advisor
        # TPE only supports single objective without constraints
        if num_objectives != 1 or num_constraints != 0:
            raise ValueError(
                f"TPE advisor only supports single objective without constraints, "
                f"but got num_objectives={num_objectives}, num_constraints={num_constraints}"
            )
        return TPE_Advisor(
            config_space,
            task_id=task_id,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )
    
    elif advisor_type == 'ea':
        from openbox.core.ea_advisor import EA_Advisor
        # EA only supports single objective without constraints
        if num_objectives != 1 or num_constraints != 0:
            raise ValueError(
                f"EA advisor only supports single objective without constraints, "
                f"but got num_objectives={num_objectives}, num_constraints={num_constraints}"
            )
        return EA_Advisor(
            config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            optimization_strategy=optimization_strategy,
            batch_size=batch_size if batch_size is not None else 1,
            task_id=task_id,
            output_dir=output_dir,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )
    
    elif advisor_type == 'random':
        from openbox.core.random_advisor import RandomAdvisor
        return RandomAdvisor(
            config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            ref_point=ref_point,
            task_id=task_id,
            output_dir=output_dir,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )
    
    elif advisor_type == 'sync_batch':
        from openbox.core.sync_batch_advisor import SyncBatchAdvisor
        return SyncBatchAdvisor(
            config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            batch_size=batch_size,
            batch_strategy=batch_strategy,
            initial_trials=initial_trials,
            initial_configurations=initial_configurations,
            init_strategy=init_strategy,
            transfer_learning_history=transfer_learning_history,
            optimization_strategy=optimization_strategy,
            surrogate_type=surrogate_type,
            acq_type=acq_type,
            acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point,
            task_id=task_id,
            output_dir=output_dir,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )
    
    elif advisor_type == 'async_batch':
        from openbox.core.async_batch_advisor import AsyncBatchAdvisor
        return AsyncBatchAdvisor(
            config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            batch_size=batch_size,
            batch_strategy=batch_strategy,
            initial_trials=initial_trials,
            initial_configurations=initial_configurations,
            init_strategy=init_strategy,
            transfer_learning_history=transfer_learning_history,
            optimization_strategy=optimization_strategy,
            surrogate_type=surrogate_type,
            acq_type=acq_type,
            acq_optimizer_type=acq_optimizer_type,
            ref_point=ref_point,
            task_id=task_id,
            output_dir=output_dir,
            random_state=random_state,
            logger_kwargs=_logger_kwargs,
            **advisor_kwargs
        )
    
    else:
        raise ValueError(
            f"Invalid advisor type: '{advisor_type}'. "
            f"Supported types: 'default', 'mcadvisor', 'tpe', 'ea', 'random', "
            f"'sync_batch', 'async_batch', 'mf'"
        )

def create_parallel_advisor(parallel_strategy, sample_strategy, **kwargs):
    """Create a parallel advisor factory function."""
    if parallel_strategy not in ['sync', 'async']:
        raise ValueError('Invalid parallel strategy: %s' % parallel_strategy)
    if sample_strategy not in ['random', 'bo', 'ea']:
        raise ValueError('Invalid sample strategy: %s' % sample_strategy)
    if parallel_strategy == 'sync':
        advisor_type = 'sync_batch' if sample_strategy in ['random', 'bo'] else 'ea'
    elif parallel_strategy == 'async':
        advisor_type = 'async_batch' if sample_strategy in ['random', 'bo'] else 'ea'
    else:
        raise ValueError(f'Invalid parallel strategy: {parallel_strategy}')
    
    return build_advisor(advisor_type=advisor_type, **kwargs)

# Register advisors (for future extensibility)
# Users can register custom advisors like:
# register_advisor('my_advisor', MyAdvisorClass, {'num_objectives': 1})