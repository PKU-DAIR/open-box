# License: MIT
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(
        y, cy=None,
        true_minimum=None, name=None, clip_y=True,
        title="Convergence plot",
        xlabel="Iteration",
        ylabel="Min objective value",
        ax=None, alpha=0.3, yscale=None,
        color='C0', infeasible_color='C1',
        **kwargs):
    """
    Plot convergence trace.

    Parameters
    ----------
    y : np.ndarray
        Objective values. y.ndim must be 1.
    cy : np.ndarray, optional
        Constraint values. cy.ndim must be 2.
    true_minimum : float, optional
        True minimum value of the objective function.
    name : str, optional
        Name of the plotted method.
    clip_y : bool, default=True
        Auto clip max y value.
    title : str
        Title of the plot.
    xlabel : str
        Label of x-axis.
    ylabel : str
        Label of y-axis.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes on which to draw the plot, or `None` to create a new one.
    alpha : float, default=0.3
        Alpha value of the scatter plot.
    yscale : str, optional
        Scale of y-axis.
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
    if ax is None:
        ax = plt.gca()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    y = np.asarray(y)
    assert y.ndim == 1, 'plot_convergence only supports single objective'
    x = np.arange(y.shape[0]) + 1

    has_constraints = cy is not None
    feasible = np.ones(y.shape[0], dtype=bool)
    if has_constraints:
        cy = np.asarray(cy)
        assert cy.ndim == 2 and cy.shape[0] == y.shape[0]
        feasible = np.all(cy <= 0, axis=1)

    has_feasible = np.any(feasible)
    x_feasible = x[feasible]
    x_infeasible = x[~feasible]
    y_feasible = y[feasible]
    y_infeasible = y[~feasible]
    y_mins = np.minimum.accumulate(y_feasible)

    if clip_y and has_feasible:
        max_y = np.max(y_mins)
        y_mins = np.clip(y_mins, None, max_y)
        y_feasible = np.clip(y_feasible, None, max_y)
        y_infeasible = np.clip(y_infeasible, None, max_y)

    # plot curve
    x_, y_ = x_feasible, y_mins
    if len(x_) > 0 and x_[-1] != x[-1]:  # extend to the last iteration
        x_ = np.concatenate([x_, [x[-1]]])
        y_ = np.concatenate([y_, [y_[-1]]])
    ax.plot(x_, y_, label=name, color=color, **kwargs)

    # plot feasible points
    y_label = "Feasible" if has_constraints else None
    ax.scatter(x_feasible, y_feasible, c=color, label=y_label, marker='o', alpha=alpha)

    # plot infeasible points
    if has_constraints:
        ax.scatter(x_infeasible, y_infeasible, c=infeasible_color, label="Infeasible", marker='^', alpha=alpha)

    # plot true minimum
    if true_minimum is not None:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    # plot legend
    if true_minimum is not None or name is not None or has_constraints:
        ax.legend(loc="upper right")
    return ax
