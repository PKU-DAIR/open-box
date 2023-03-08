import abc
from typing import Union
from openbox import History


def build_visualizer(
        option: Union[str, bool],
        history: History,
        *,
        logging_dir='logs/',
        task_info=None,
        optimizer=None,
        advisor=None,
        **kwargs,
):
    """
    Build visualizer for optimizer.

    Parameters
    ----------
    option : ['none', 'basic', 'advanced']
        Visualizer option.
    history : History
        History to visualize.
    logging_dir : str, default='logs/'
        The directory to save the visualization.
    task_info : dict, optional
        Task information for visualizer to use.
    optimizer : Optimizer, optional
        Optimizer to extract task_info from.
    advisor : Advisor, optional
        Advisor to extract task_info from.
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
        if task_info is None:
            task_info = dict()
        _task_info = extract_task_info(optimizer=optimizer, advisor=advisor)
        _task_info.update(task_info)

        from openbox.visualization.html_visualizer import HTMLVisualizer
        visualizer = HTMLVisualizer(
            logging_dir=logging_dir,
            history=history,
            task_info=_task_info,
            auto_open_html=kwargs.get('auto_open_html', False),
            advanced_analysis=(option == 'advanced'),
            advanced_analysis_options=kwargs.get('advanced_analysis_options'),
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


def extract_task_info(*, optimizer=None, advisor=None):
    """
    Extract task information from optimizer or advisor.

    Parameters
    ----------
    optimizer : Optimizer, optional
        Optimizer to extract task_info from.

    advisor : Advisor, optional
        Advisor to extract task_info from.

    Returns
    -------
    task_info : dict
        Task information for visualizer to use.
    """
    if optimizer is None and advisor is None:
        return dict()

    if advisor is None:
        advisor = optimizer.config_advisor if hasattr(optimizer, 'config_advisor') else None

    task_info = dict()

    if optimizer is not None:
        task_info.update({
            'advisor_type': optimizer.advisor_type,
            'max_runs': optimizer.max_runs,
            'max_runtime_per_trial': optimizer.max_runtime_per_trial,
        })
    if advisor is not None:
        task_info.update({
            # todo: if model is altered, this will not be updated
            'surrogate_type': advisor.surrogate_type if hasattr(advisor, 'surrogate_type') else None,
            'constraint_surrogate_type': advisor.constraint_surrogate_type if hasattr(
                advisor, 'constraint_surrogate_type') else None,
            'transfer_learning_history': advisor.transfer_learning_history if hasattr(
                advisor, 'transfer_learning_history') else None,
        })
    return task_info


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
