# License: MIT
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(
        x,
        y,
        name=None,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale=None,
        yscale=None,
        grid=True,
        ax=None,
        **kwargs):
    """
    Plot a curve.

    Parameters
    ----------
    x : array_like
        x-axis data.
    y : array_like
        y-axis data.
    name : str, optional
        Name of the curve.
    title : str, optional
        Title of the plot.
    xlabel, ylabel : str, optional
        Label of x-axis and y-axis.
    xlim, ylim : tuple, optional
        Limit of x-axis and y-axis.
    xscale, yscale : str, optional
        Scale of x-axis and y-axis.
    grid : bool, default=True
        Whether to show grid.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes on which to draw the plot, or `None` to create a new one.
    kwargs : dict
        Other keyword arguments passed to `ax.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    x = np.asarray(x)
    y = np.asarray(y)
    ax.plot(x, y, label=name, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(grid)

    # plot legend
    if name is not None:
        ax.legend(loc="upper right")
    return ax
