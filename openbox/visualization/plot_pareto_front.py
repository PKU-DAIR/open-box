# License: MIT
import numpy as np
import matplotlib.pyplot as plt
from openbox.utils.multi_objective import get_pareto_front


def plot_pareto_front(
        y, cy=None,
        title="Pareto Front",
        ax=None, alpha=0.3,
        color='C0', infeasible_color='C1',
        **kwargs):
    """
    Plot Pareto front for multi-objective optimization.

    Parameters
    ----------
    y : np.ndarray
        Objective values. Shape: (n_configs, num_objectives).
        num_objectives must be 2 or 3. Larger num_objectives is not supported.
    cy : np.ndarray, optional
        Constraint values. Shape: (n_configs, num_constraints).
    title : str
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes on which to draw the plot, or `None` to create a new one.
    alpha : float, default=0.3
        Alpha value of the scatter plot.
    color : str, default='C0' (first color in default color cycle)
        Color of the curve and feasible points.
    infeasible_color : str, default='C1' (second color in default color cycle)
        Color of the infeasible points.
    kwargs : dict
        Other keyword arguments passed to `ax.plot` or `ax.plot_trisurf`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes.
    """
    y = np.asarray(y)
    assert y.ndim == 2, 'plot_pareto_front only supports multi-objective optimization.'
    num_objectives = y.shape[1]
    assert num_objectives in [2, 3], 'plot_pareto_front only supports 2 or 3 objectives.'

    has_constraints = cy is not None
    feasible = np.ones(y.shape[0], dtype=bool)
    if has_constraints:
        cy = np.asarray(cy)
        assert cy.ndim == 2 and cy.shape[0] == y.shape[0]
        feasible = np.all(cy <= 0, axis=1)

    has_feasible = np.any(feasible)
    # y_feasible and y_infeasible may have shape (0, num_objectives)!
    y_feasible = y[feasible]
    y_infeasible = y[~feasible]

    if num_objectives == 2:
        ax = _plot_pareto_front_2d(
            y_feasible=y_feasible, y_infeasible=y_infeasible,
            ax=ax, alpha=alpha,
            color=color, infeasible_color=infeasible_color,
            **kwargs,
        )
    elif num_objectives == 3:
        ax = _plot_pareto_front_3d(
            y_feasible=y_feasible, y_infeasible=y_infeasible,
            ax=ax, alpha=alpha,
            color=color, infeasible_color=infeasible_color,
            **kwargs,
        )

    ax.set_title(title)

    # plot legend
    if has_constraints:
        ax.legend(loc="upper right")
    return ax


def _plot_pareto_front_2d(
        y_feasible=None, y_infeasible=None,
        ax=None, alpha=0.3,
        color='C0', infeasible_color='C1',
        **kwargs,
):
    """
    Plot 2-D Pareto front for multi-objective optimization.

    Parameters
    ----------
    y_feasible : np.ndarray, optional
        Feasible objective values. Shape: (n_f_samples, 2).
    y_infeasible : np.ndarray, optional
        Infeasible objective values. Shape: (n_inf_samples, 2).
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes on which to draw the plot, or `None` to create a new one.
    alpha : float, default=0.3
        Alpha value of the scatter plot.
    color : str, default='C0' (first color in default color cycle)
        Color of the curve and feasible points.
    infeasible_color : str, default='C1' (second color in default color cycle)
        Color of the infeasible points.
    kwargs : dict
        Other keyword arguments passed to `ax.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes.
    """
    if y_feasible is not None:
        assert y_feasible.ndim == 2 and y_feasible.shape[1] == 2
    if y_infeasible is not None:
        assert y_infeasible.ndim == 2 and y_infeasible.shape[1] == 2

    if ax is None:
        ax = plt.gca()

    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.grid()
    # plot points
    if y_feasible is not None:
        ax.scatter(y_feasible[:, 0], y_feasible[:, 1],
                   c=color, label="Feasible", marker='o', alpha=alpha)
    if y_infeasible is not None:
        ax.scatter(y_infeasible[:, 0], y_infeasible[:, 1],
                   c=infeasible_color, label="Infeasible", marker='^', alpha=alpha)

    if y_feasible is not None:
        pareto_front = get_pareto_front(y_feasible)
        # sort pareto front
        pareto_front = pareto_front[np.lexsort(pareto_front.T)]
        # plot pareto front
        ax.plot(pareto_front[:, 0], pareto_front[:, 1], c=color, **kwargs)
    return ax


def _plot_pareto_front_3d(
        y_feasible=None, y_infeasible=None,
        ax=None, alpha=0.3,
        color='C0', infeasible_color='C1',
        **kwargs,
):
    """
    Plot 3-D Pareto front for multi-objective optimization.

    Parameters
    ----------
    y_feasible : np.ndarray, optional
        Feasible objective values. Shape: (n_f_samples, 3).
    y_infeasible : np.ndarray, optional
        Infeasible objective values. Shape: (n_inf_samples, 3).
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes on which to draw the plot, or `None` to create a new one.
    alpha : float, default=0.3
        Alpha value of the scatter plot.
    color : str, default='C0' (first color in default color cycle)
        Color of the curve and feasible points.
    infeasible_color : str, default='C1' (second color in default color cycle)
        Color of the infeasible points.
    kwargs : dict
        Other keyword arguments passed to `ax.plot_trisurf`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes.
    """
    if y_feasible is not None:
        assert y_feasible.ndim == 2 and y_feasible.shape[1] == 3
    if y_infeasible is not None:
        assert y_infeasible.ndim == 2 and y_infeasible.shape[1] == 3

    if ax is None:
        ax = plt.axes(projection='3d')

    ax.set_xlabel('Objective 1')
    ax.set_ylabel('Objective 2')
    ax.set_zlabel('Objective 3')
    ax.grid()
    ax.invert_xaxis()

    # plot points
    if y_feasible is not None:
        ax.scatter(y_feasible[:, 0], y_feasible[:, 1], y_feasible[:, 2],
                   c=color, label="Feasible", marker='o', alpha=alpha)
    if y_infeasible is not None:
        ax.scatter(y_infeasible[:, 0], y_infeasible[:, 1], y_infeasible[:, 2],
                   c=infeasible_color, label="Infeasible", marker='^', alpha=alpha)

    if y_feasible is not None:
        pareto_front = get_pareto_front(y_feasible)
        # sort pareto front  # todo: check if this is necessary
        pareto_front = pareto_front[np.lexsort(pareto_front.T)]
        # plot pareto front (3d surface)
        ax.plot_trisurf(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                        color=color, alpha=alpha, **kwargs)
    return ax
