# License: MIT

"""
https://github.com/automl/fanova/blob/master/fanova/visualizer.py
"""
import itertools as it
import logging
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace.hyperparameters import Hyperparameter, CategoricalHyperparameter, Constant, OrdinalHyperparameter, \
    NumericalHyperparameter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Visualizer(object):

    def __init__(self, fanova, cs, directory, y_label='Performance'):
        """
        Parameters
        ------------
        fanova:fANOVA object
        cs: ConfigSpace instantiation
        directory: str
            Path to the directory in which all plots will be stored
        """
        self.fanova = fanova
        self.cs = cs
        self.cs_params = cs.get_hyperparameters()
        if not os.path.exists(directory):
            raise FileNotFoundError("Directory %s doesn't exist." % directory)
        self.directory = directory
        self._y_label = y_label
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

    def create_all_plots(self, three_d=True, **kwargs):
        """
        Creates plots for all main effects and stores them into a directory
        Specifically, all single and pairwise marginals are plotted

        Parameters
        ----------
        three_d: boolean
            whether or not to plot pairwise marginals in 3D-plot
        """
        # single marginals
        for param_idx in range(len(self.cs_params)):
            param_name = self.cs_params[param_idx].name
            plt.close()
            outfile_name = os.path.join(self.directory, param_name.replace(os.sep, "_") + ".png")
            self.logger.info("creating %s" % outfile_name)

            self.plot_marginal(param_idx, show=False, **kwargs)
            plt.savefig(outfile_name)

        # additional pairwise plots:
        dimensions = list(range(len(self.cs_params)))
        combis = list(it.combinations(dimensions, 2))
        for combi in combis:
            param_names = [self.cs_params[p].name for p in combi]
            plt.close()
            param_names = str(param_names)
            param_names = re.sub('[!,@#\'\n$\[\]]', '', param_names)
            outfile_name = os.path.join(self.directory, str(param_names).replace(" ", "_") + ".png")
            self.logger.info("creating %s" % outfile_name)
            self.plot_pairwise_marginal(combi, three_d=three_d, **kwargs)
            plt.savefig(outfile_name)

    def generate_pairwise_marginal(self, param_list, resolution=20):
        """
        Creates a plot of pairwise marginal of a selected parameters

        Parameters
        ----------
        param_list: list of ints or strings
            Contains the selected parameters
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        Returns
        -------
        grid_orig: List[List[...]]
            nested list of two lists with the values that were evaluated for parameters 1 and 2 respectively
        zz: np.array
            array that contains the predicted importance (shape is len(categoricals) or resolution depending on
            parameter, so e.g. zz = np.array(3, 20) if first parameter has three categories and resolution is set to 20
        """
        if len(set(param_list)) != 2:
            raise ValueError("You have to specify 2 (different) parameters")

        params, param_names, param_indices = self._get_parameter(param_list)

        # The grid_fanova contains only numeric values (even for categorical choices) while grid_orig contains the
        #   actual value (also for categorical choices!)
        grid_fanova, grid_orig = [], []

        for p in params:
            if isinstance(p, CategoricalHyperparameter):
                grid_orig.append(p.choices)
                grid_fanova.append(np.arange(len(p.choices)))
            elif isinstance(p, OrdinalHyperparameter):
                grid_orig.append(p.sequence)
                grid_fanova.append(np.arange(len(p.sequence)))
            elif isinstance(p, Constant):
                grid_orig.append((p.value,))
                grid_fanova.append(np.arange(1))
            elif isinstance(p, NumericalHyperparameter):
                if p.log:
                    base = np.e  # assuming ConfigSpace uses the natural logarithm
                    log_lower = np.log(p.lower) / np.log(base)
                    log_upper = np.log(p.upper) / np.log(base)
                    grid = np.logspace(log_lower, log_upper, resolution, endpoint=True, base=base)
                else:
                    grid = np.linspace(p.lower, p.upper, resolution)
                grid_orig.append(grid)
                grid_fanova.append(grid)
            else:
                raise ValueError("Hyperparameter %s of type %s not supported." % (p.name, type(p)))

        # Turn into arrays, squeeze all but the first two dimensions (avoid squeezing away the dimension for Constants)
        param_indices = np.array(param_indices)
        param_indices = param_indices.reshape([s for i, s in enumerate(param_indices.shape) if i in [0, 1] or s != 1])
        grid_fanova = np.array(grid_fanova)
        grid_fanova = grid_fanova.reshape([s for i, s in enumerate(grid_fanova.shape) if i in [0, 1] or s != 1])

        # Populating the result
        zz = np.zeros((len(grid_fanova[0]), len(grid_fanova[1])))
        for i, x_value in enumerate(grid_fanova[0]):
            for j, y_value in enumerate(grid_fanova[1]):
                zz[i][j] = self.fanova.marginal_mean_variance_for_values(param_indices, [x_value, y_value])[0]

        return grid_orig, zz

    def plot_pairwise_marginal(self, param_list, resolution=20, show=False, three_d=True, colormap=cm.jet,
                               add_colorbar=True):
        """
        Creates a plot of pairwise marginal of a selected parameters

        Parameters
        ----------
        param_list: list of ints or strings
            Contains the selected parameters
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict
        show: boolean
            whether to call plt.show() to show plot directly as interactive matplotlib-plot
        three_d: boolean
            whether or not to plot pairwise marginals in 3D-plot
        colormap: matplotlib.Colormap
            which colormap to use for the 3D plots
        add_colorbar: bool
            whether to add the colorbar for 3d plots
        """
        if len(set(param_list)) != 2:
            raise ValueError("You have to specify 2 (different) parameters")

        params, param_names, param_indices = self._get_parameter(param_list)

        first_is_numerical = isinstance(params[0], NumericalHyperparameter)
        second_is_numerical = isinstance(params[1], NumericalHyperparameter)

        plt.close()
        fig = plt.figure()
        plt.title('%s and %s' % (param_names[0], param_names[1]))

        if first_is_numerical and second_is_numerical:
            # No categoricals -> create heatmap / 3D-plot
            grid_list, zz = self.generate_pairwise_marginal(param_indices, resolution)

            z_min, z_max = zz.min(), zz.max()
            display_xx, display_yy = np.meshgrid(grid_list[0], grid_list[1])

            if three_d:
                ax = Axes3D(fig)
                surface = ax.plot_surface(display_xx, display_yy, zz.T,
                                          rstride=1, cstride=1, cmap=colormap, linewidth=0, antialiased=False)
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
                ax.set_zlabel(self._y_label)
                if add_colorbar:
                    fig.colorbar(surface, shrink=0.5, aspect=5)

            else:
                plt.pcolor(display_xx, display_yy, zz.T, cmap=colormap, vmin=z_min, vmax=z_max)
                plt.xlabel(param_names[0])

                if self.cs_params[param_indices[0]].log:
                    plt.xscale('log')
                if self.cs_params[param_indices[1]].log:
                    plt.yscale('log')

                plt.ylabel(param_names[1])
                plt.colorbar()
        else:
            # At least one of the two parameters is non-numerical (categorical, ordinal or constant)
            if first_is_numerical or second_is_numerical:
                # Only one of them is non-numerical -> create multi-line-plot
                # Make sure categorical is first in indices (for iteration below)
                numerical_idx = 0 if first_is_numerical else 1
                categorical_idx = 1 - numerical_idx
                grid_labels, zz = self.generate_pairwise_marginal(param_indices, resolution)

                if first_is_numerical:
                    zz = zz.T

                for i, cat in enumerate(grid_labels[categorical_idx]):
                    if params[numerical_idx].log:
                        plt.semilogx(grid_labels[numerical_idx], zz[i], label='%s' % str(cat))
                    else:
                        plt.plot(grid_labels[numerical_idx], zz[i], label='%s' % str(cat))

                plt.ylabel(self._y_label)
                plt.xlabel(param_names[numerical_idx])  # x-axis displays numerical
                plt.legend()
                plt.tight_layout()

            else:
                # Both parameters are categorical -> create hotmap
                choices, zz = self.generate_pairwise_marginal(param_indices, resolution)
                plt.imshow(zz.T, cmap='hot', interpolation='nearest')
                plt.xticks(np.arange(0, len(choices[0])), choices[0], fontsize=8)
                plt.yticks(np.arange(0, len(choices[1])), choices[1], fontsize=8)
                plt.xlabel(param_names[0])
                plt.ylabel(param_names[1])
                plt.colorbar().set_label(self._y_label)

        if show:
            plt.show()
        else:
            interact_dir = os.path.join(self.directory, 'interactive_plots')
            if not os.path.exists(interact_dir):
                self.logger.info('creating %s' % interact_dir)
                os.makedirs(interact_dir)
            try:
                pickle.dump(fig, open(interact_dir + '/%s_%s.fig.pkl' % (param_names[0], param_names[1]), 'wb'))
            except AttributeError as err:
                self.logger.debug(err, exc_info=True)
                self.logger.info("Pickling the interactive pairwise-marginal plot (%s) raised an exception. Resume "
                                 "without pickling. ", str(param_names))

        return plt

    def generate_marginal(self, p, resolution=100):
        """
        Creates marginals of a selected parameter for own plots

        Parameters
        ----------
        p: int or str
            Index of chosen parameter in the ConfigSpace (starts with 0) or name
        resolution: int
            Number of samples to generate from the parameter range as
            values to predict

        """
        p, p_name, p_idx = self._get_parameter(p)

        if isinstance(p, NumericalHyperparameter):
            lower_bound = p.lower
            upper_bound = p.upper
            log = p.log
            if log:
                base = np.e  # assuming ConfigSpace uses the natural logarithm
                log_lower = np.log(lower_bound) / np.log(base)
                log_upper = np.log(upper_bound) / np.log(base)
                grid = np.logspace(log_lower, log_upper, resolution, endpoint=True, base=base)

                if abs(grid[0] - lower_bound) > 0.00001:
                    self.logger.warning("Check the grid's (lower) accuracy for %s (plotted vs theoretical: %s vs %s)"
                                        % (p.name, grid[0], lower_bound))
                if abs(grid[-1] - upper_bound) > 0.00001:
                    self.logger.warning("Check the grid's (upper) accuracy for %s (plotted vs theoretical: %s vs %s)"
                                        % (p.name, grid[-1], upper_bound))

            else:
                grid = np.linspace(lower_bound, upper_bound, resolution)
            mean = np.zeros(resolution)
            std = np.zeros(resolution)

            dim = [p_idx]
            for i in range(0, resolution):
                (m, v) = self.fanova.marginal_mean_variance_for_values(dim, [grid[i]])
                mean[i] = m
                std[i] = np.sqrt(v)
            return mean, std, grid

        else:
            if isinstance(p, CategoricalHyperparameter):
                categorical_size = len(p.choices)
            elif isinstance(p, Constant):
                categorical_size = 1
            elif isinstance(p, OrdinalHyperparameter):
                categorical_size = len(p.sequence)
            else:
                raise ValueError("Parameter %s of type %s not supported." % (p.name, type(p)))
            marginals = [self.fanova.marginal_mean_variance_for_values([p_idx], [i]) for i in range(categorical_size)]
            mean, v = list(zip(*marginals))
            std = np.sqrt(v)
            return mean, std

    def plot_marginal(self, param, resolution=100, log_scale=None, show=True, incumbents=None):
        """
        Creates a plot of marginal of a selected parameter

        Parameters
        ----------
        param: int or str
            Index of chosen parameter in the ConfigSpace (starts with 0)
        resolution: int
            Number of samples to generate from the parameter range as values to predict
        log_scale: boolean
            If log scale is required or not. If no value is given, it is deduced from the ConfigSpace provided
        show: boolean
            whether to call plt.show() to show plot directly as interactive matplotlib-plot
        incumbents: List[Configuration]
            list of ConfigSpace.Configurations that are marked as incumbents
        """
        param, param_name, param_idx = self._get_parameter(param)

        # check if categorical
        if isinstance(param, NumericalHyperparameter):
            # PREPROCESS
            mean, std, grid = self.generate_marginal(param_idx, resolution)
            mean = np.asarray(mean)
            std = np.asarray(std)

            lower_curve = mean - std
            upper_curve = mean + std

            if log_scale is None:
                log_scale = param.log or (np.diff(grid).std() > 0.000001)

            # PLOT
            if log_scale:
                if np.diff(grid).std() > 0.000001:
                    self.logger.info("It might be better to plot this parameter '%s' in log-scale.", param_name)
                plt.semilogx(grid, mean, 'b', label='predicted %s' % self._y_label)
            else:
                plt.plot(grid, mean, 'b', label='predicted %s' % self._y_label)
            plt.fill_between(grid, upper_curve, lower_curve, facecolor='red', alpha=0.6, label='std')

            if incumbents is not None:
                if not isinstance(incumbents, list):
                    incumbents = [incumbents]
                values = [inc[param_name] for inc in incumbents if param_name in inc and inc[param_name] is not None]
                indices = [(np.abs(np.asarray(grid) - val)).argmin() for val in values]
                if len(indices) > 0:
                    plt.scatter(list([grid[idx] for idx in indices]),
                                list([mean[idx] for idx in indices]),
                                label='incumbent', c='black', marker='.', zorder=999)

            plt.xlabel(param_name)
            plt.ylabel(self._y_label)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

        else:
            # PREPROCESS
            if isinstance(param, CategoricalHyperparameter):
                labels = param.choices
                categorical_size = len(param.choices)
            elif isinstance(param, OrdinalHyperparameter):
                labels = param.sequence
                categorical_size = len(param.sequence)
            elif isinstance(param, Constant):
                labels = str(param)
                categorical_size = 1
            else:
                raise ValueError("Parameter %s of type %s not supported." % (param.name, type(param)))

            indices = np.arange(1, categorical_size + 1, 1)
            mean, std = self.generate_marginal(param_idx)
            min_y = mean[0]
            max_y = mean[0]

            # PLOT
            b = plt.boxplot([[x] for x in mean])
            plt.xticks(indices, labels)
            # blow up boxes
            for box, std_ in zip(b["boxes"], std):
                y = box.get_ydata()
                y[2:4] = y[2:4] + std_
                y[0:2] = y[0:2] - std_
                y[4] = y[4] - std_
                box.set_ydata(y)
                min_y = min(min_y, y[0] - std_)
                max_y = max(max_y, y[2] + std_)

            plt.ylim([min_y, max_y])

            plt.ylabel(self._y_label)
            plt.xlabel(param_name)
            plt.tight_layout()

        if show:
            plt.show()
        else:
            return plt

    def create_most_important_pairwise_marginal_plots(self, params=None, n=20, three_d=True, resolution=20):
        """
        Creates plots of the n most important pairwise marginals of the whole ConfigSpace

        Parameters
        ------------
        params: list
             Contains the selected parameters for pairwise evaluation
        n: int
             The number of most relevant pairwise marginals that will be returned
        three_d: boolean
            whether or not to plot pairwise marginals in 3D-plot
        resolution: int
            number of values to be evaluated for non-categoricals
        """
        if self.fanova._dict:
            most_important_pairwise_marginals = self.fanova.tot_imp_dict
        else:
            if params is not None:
                most_important_pairwise_marginals = self.fanova.get_most_important_pairwise_marginals(params=params)
            else:
                most_important_pairwise_marginals = self.fanova.get_most_important_pairwise_marginals(n=n)

        for param1, param2 in most_important_pairwise_marginals:
            params, param_names, param_indices = self._get_parameter([param1, param2])
            param_names_str = re.sub('[!,@#\'\n$\[\]]', '', str(param_names))
            outfile_name = os.path.join(self.directory, str(param_names_str).replace(" ", "_") + ".png")
            self.logger.info("creating %s" % outfile_name)
            self.plot_pairwise_marginal((param1, param2), show=False, three_d=three_d, resolution=resolution)
            plt.savefig(outfile_name)

    def _get_parameter(self, orig_p):
        """
        Allows for arbitrary access to parameter(s) p
        
        Parameters
        ----------
        orig_p: String or Int or Hyperparameter or List of those
            either representation of a hyperparameter
        
        Returns
        -------
        p, p_name, p_idx:
            All three representations 
        """
        p_list, p_names, p_indices = [], [], []

        if not (isinstance(orig_p, list) or isinstance(orig_p, tuple)):
            orig_p = [orig_p]
        for p in orig_p:
            if isinstance(p, Hyperparameter):
                p_idx = self.cs.get_idx_by_hyperparameter_name(p.name)
            elif isinstance(p, str):
                p_idx = self.cs.get_idx_by_hyperparameter_name(p)
            elif isinstance(p, int):
                p_idx = p
            else:
                raise ValueError("{} (type: {}) not a interpretable as a parameter!".format(p, type(p)))
            p_list.append(self.cs_params[p_idx])
            p_names.append(self.cs_params[p_idx].name)
            p_indices.append(p_idx)

        if len(orig_p) == 1:
            return p_list[0], p_names[0], p_indices[0]
        else:
            return p_list, p_names, p_indices
