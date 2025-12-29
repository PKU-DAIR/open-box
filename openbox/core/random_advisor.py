# License: MIT

from openbox.core.generic_advisor import Advisor
from openbox.utils.util_funcs import deprecate_kwarg


class RandomAdvisor(Advisor):
    """
    Random Advisor Class, which adopts the random policy to sample a configuration.
    
    This is a convenience wrapper around Advisor with optimization_strategy='random'.
    It supports initial configurations via InitialConfigProvider before falling back 
    to random sampling.
    
    Parameters
    ----------
    config_space : openbox.space.Space or ConfigSpace.ConfigurationSpace
        Configuration space.
    num_objectives : int, default=1
        Number of objectives in objective function.
    num_constraints : int, default=0
        Number of constraints in objective function.
    initial_trials : int, default=0
        Number of initial iterations before random sampling.
        Set to 0 for pure random search without initial design.
    init_strategy : str, default='random'
        Strategy to generate configurations for initial iterations.
    initial_configurations : List[Configuration], optional
        User-provided initial configurations.
    ref_point : List[float], optional
        Reference point for multi-objective optimization.
    early_stop : bool, default=False
        Whether to enable early stop.
    early_stop_kwargs : dict, optional
        Options for early stop algorithm.
    output_dir : str, default='logs'
        Directory to save log files.
    task_id : str, default='OpenBox'
        Task identifier.
    random_state : int, optional
        Random seed for RNG.
    logger_kwargs : dict, optional
        Additional keyword arguments for logger.
    """

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            initial_trials=0,
            init_strategy='random',
            initial_configurations=None,
            ref_point=None,
            early_stop=False,
            early_stop_kwargs=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
    ):
        super().__init__(
            config_space=config_space,
            num_objectives=num_objectives,
            num_constraints=num_constraints,
            initial_trials=initial_trials,
            init_strategy=init_strategy,
            initial_configurations=initial_configurations,
            rand_prob=1.0,  # Always sample random after initial trials
            optimization_strategy='random',  # Use random strategy
            ref_point=ref_point,
            early_stop=early_stop,
            early_stop_kwargs=early_stop_kwargs,
            output_dir=output_dir,
            task_id=task_id,
            random_state=random_state,
            logger_kwargs=logger_kwargs,
        )
