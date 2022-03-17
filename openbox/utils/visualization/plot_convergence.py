# License: MIT

# Thanks to :
# @Author  : qichun tang
# @Date    : 2020-12-27
# @Contact    : qichun.tang@bupt.edu.cn
import matplotlib.pyplot as plt


def plot_convergence(
        x, y1, y2,
        xlabel="Number of iterations $n$",
        ylabel=r"Min objective value after $n$ iterations",
        ax=None, name=None, alpha=0.2, yscale=None,
        color=None, true_minimum=None,
        **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    args[i] :  `OptimizeResult`, list of `OptimizeResult`, or tuple
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    ax : `Axes`, optional
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    true_minimum : float, optional
        The true minimum value of the function, if known.

    yscale : None or string, optional
        The scale for the y-axis.

    Returns
    -------
    ax : `Axes`
        The matplotlib axes.
    """
    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.plot(x, y1, c=color, label=name, **kwargs)
    ax.scatter(x, y2, c=color, alpha=alpha)

    if true_minimum is not None:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum is not None or name is not None:
        ax.legend(loc="upper right")
    return ax
