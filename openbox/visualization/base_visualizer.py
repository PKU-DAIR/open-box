import abc
from typing import Union


def build_visualizer(option: Union[str, bool], optimizer, **kwargs):
    """
    Build visualizer for optimizer.

    Parameters
    ----------
    option : ['none', 'basic', 'advanced']
        Visualizer option.

    optimizer : Optimizer
        Optimizer to visualize.

    kwargs : dict
        Other arguments for visualizer.
        For HTMLVisualizer, available arguments are:
        - auto_open_html : bool, default=False
            Whether to open html file automatically.
        - advanced_analysis_options : dict, default=None
            Advanced analysis options. See `HTMLVisualizer` for details.

    Returns
    -------
    visualizer : BaseVisualizer
        Visualizer.
    """
    option = _parse_option(option)

    if option == 'none':
        visualizer = NullVisualizer()
    elif option in ['basic', 'advanced']:
        advisor = optimizer.config_advisor
        from openbox.visualization.html_visualizer import HTMLVisualizer
        visualizer = HTMLVisualizer(
            logging_dir=optimizer.output_dir,
            history=optimizer.get_history(),
            auto_open_html=kwargs.get('auto_open_html', False),
            advanced_analysis=(option == 'advanced'),
            advanced_analysis_options=kwargs.get('advanced_analysis_options'),
            advisor_type=optimizer.advisor_type,
            surrogate_type=advisor.surrogate_type if hasattr(advisor, 'surrogate_type') else None,
            max_iterations=optimizer.max_iterations,
            time_limit_per_trial=optimizer.time_limit_per_trial,
            surrogate_model=advisor.surrogate_model if hasattr(advisor, 'surrogate_model') else None,
            constraint_models=advisor.constraint_models if hasattr(advisor, 'constraint_models') else None,
        )
    else:
        raise ValueError('Unknown visualizer option: %s' % option)

    return visualizer


def _parse_option(option: Union[str, bool]):
    if isinstance(option, str):
        option = option.lower()
    else:
        if not option:  # None, False, 0
            option = 'none'
        else:
            option = 'basic'

    assert option in ['none', 'basic', 'advanced']
    return option


class BaseVisualizer(object, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def visualize(self, *args, **kwargs):
        raise NotImplementedError


class NullVisualizer(BaseVisualizer):
    """
    Do not visualize anything.
    """
    def setup(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def visualize(self, *args, **kwargs):
        pass
