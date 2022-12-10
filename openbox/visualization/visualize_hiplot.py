import os
import numpy as np
from typing import List, Optional
from ConfigSpace import Configuration
from openbox import logger


def visualize_hiplot(
        configs: List[Configuration],
        y: np.ndarray,
        cy: np.ndarray = None,
        html_file: Optional[str] = None,
        **kwargs,
):
    """
    Visualize the history using HiPlot in Jupyter Notebook.

    HiPlot documentation: https://facebookresearch.github.io/hiplot/

    Parameters
    ----------
    configs : List[Configuration]
        List of configurations.
    y: np.ndarray
        Objective values.
    cy: np.ndarray
        Constraint values.
    html_file: str, optional
        If None, the visualization will be shown in Jupyter Notebook.
        If specified, the visualization will be saved to the html file.
    kwargs: dict
        Other keyword arguments passed to `hiplot.Experiment.display` or `hiplot.Experiment.to_html`.

    Returns
    -------
    exp: hiplot.Experiment
        The hiplot experiment object.
    """
    try:
        import hiplot
    except ModuleNotFoundError:
        logger.error("Please run 'pip install hiplot'. "
                     "HiPlot requires Python 3.6 or newer.")
        raise

    if len(configs) == 0:
        logger.error("No configurations to visualize.")
        return

    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    assert y.ndim == 2, "y.ndim must be 1 or 2."
    assert len(configs) == y.shape[0], "The length of configs must be equal to the number of rows of y."
    if cy is not None:
        cy = np.asarray(cy)
        assert cy.ndim == 2, "cy.ndim must be 2."
        assert cy.shape[0] == y.shape[0], "cy.shape[0] must be equal to y.shape[0]."

    parameters = configs[0].configuration_space.get_hyperparameter_names()
    for j in range(y.shape[1]):
        obj_name = 'Obj %d' % j
        assert obj_name not in parameters, f"The name of objective {j} conflicts with the name of hyperparameter."

    if cy is not None:
        for j in range(cy.shape[1]):
            cons_name = 'Cons %d' % j
            assert cons_name not in parameters, f"The name of constraint {j} conflicts with the name of hyperparameter."

    # todo: although dict is insertion ordered in Python 3.6+
    #   (https://stackoverflow.com/questions/39980323/are-dictionaries-ordered-in-python-3-6),
    #   the plot order is not stable.
    data = []
    for i, config in enumerate(configs):
        trial = dict()
        config_dict = config.get_dictionary()
        for param in parameters:
            trial[param] = config_dict.get(param)
        for j in range(y.shape[1]):
            trial['Obj %d' % j] = y[i, j]
        if cy is not None:
            for j in range(cy.shape[1]):
                trial['Cons %d' % j] = cy[i, j]
        data.append(trial)
    exp = hiplot.Experiment.from_iterable(data)

    if html_file is None:
        exp.display(**kwargs)
    else:
        dirname = os.path.dirname(html_file)
        if dirname != '' and not os.path.exists(dirname):
            logger.info(f'Directory "{dirname}" does not exist, create it.')
            os.makedirs(dirname, exist_ok=True)
        exp.to_html(html_file, **kwargs)
        logger.info(f"HiPlot visualization saved to {html_file}")
    return exp
