# License: MIT

"""
https://github.com/automl/fanova/blob/master/fanova/fanova.py
"""
import itertools as it
import logging
from collections import OrderedDict

import ConfigSpace
import numpy as np
import pandas as pd
import pyrfr.regression as reg
import pyrfr.util
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    NumericalHyperparameter, Constant, OrdinalHyperparameter


class fANOVA(object):
    def __init__(self, X, Y, config_space=None,
                 n_trees=16, seed=None, bootstrapping=True,
                 points_per_tree=None, max_features=None,
                 min_samples_split=0, min_samples_leaf=0,
                 max_depth=64, cutoffs=(-np.inf, np.inf)):

        """
        Calculate and provide midpoints and sizes from the forest's 
        split values in order to get the marginals
        
        Parameters
        ------------
        X: matrix with the features, either a np.array or a pd.DataFrame (numerically encoded)
        
        Y: vector with the response values (numerically encoded)
        
        config_space : ConfigSpace instantiation
        
        n_trees: number of trees in the forest to be fit
        
        seed: seed for the forests randomness
        
        bootstrapping: whether or not to bootstrap the data for each tree
        
        points_per_tree: number of points used for each tree 
                        (only subsampling if bootstrapping is false)
        
        max_features: number of features to be used at each split, default is 70%
        
        min_samples_split: minimum number of samples required to attempt to split 
        
        min_samples_leaf: minimum number of samples required in a leaf
        
        max_depth: maximal depth of each tree in the forest
        
        cutoffs: tuple of (lower, upper), all values outside this range will be
                 mapped to either the lower or the upper bound. (See:
                 "Generalized Functional ANOVA Diagnostics for High Dimensional
                 Functions of Dependent Variables" by Hooker.)
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__module__ + '.' + self.__class__.__name__)

        pcs = [(np.nan, np.nan)] * X.shape[1]

        # Convert pd.DataFrame to np.array
        if isinstance(X, pd.DataFrame):
            self.logger.debug("Detected pandas dataframes, converting to floats...")
            if config_space is not None:
                # Check if column names match parameter names
                bad_input = set(X.columns) - set(config_space.get_hyperparameter_names())
                if len(bad_input) != 0:
                    raise ValueError("Could not identify parameters %s from pandas dataframes" % str(bad_input))
                # Reorder dataframe if necessary
                X = X[config_space.get_hyperparameter_names()]
            X = X.to_numpy()
        elif config_space is not None:
            # There is a config_space but no way to check if the np.array'ed data in X is in the correct order...
            # self.logger.warning("Note that fANOVA expects data to be ordered like the return of ConfigSpace's "
            #                     "'get_hyperparameters'-method. We recommend to use labeled pandas dataframes to "
            #                     "avoid any problems.")
            pass

        # if no ConfigSpace is specified, let's build one with all continuous variables
        if config_space is None:
            # if no info is given, use min and max values of each variable as bounds
            config_space = ConfigSpace.ConfigurationSpace()
            for i, (mn, mx) in enumerate(zip(np.min(X, axis=0), np.max(X, axis=0))):
                config_space.add_hyperparameter(UniformFloatHyperparameter("x_%03i" % i, mn, mx))

        self.percentiles = np.percentile(Y, range(0, 100))
        self.cs = config_space
        self.cs_params = self.cs.get_hyperparameters()
        self.n_dims = len(self.cs_params)
        self.n_trees = n_trees
        self._dict = False

        # at this point we have a valid ConfigSpace object
        # check if param number is correct etc:
        if X.shape[1] != len(self.cs_params):
            raise RuntimeError('Number of parameters in ConfigSpace object does not match input X')
        for i in range(len(self.cs_params)):
            if isinstance(self.cs_params[i], NumericalHyperparameter):
                if (np.max(X[:, i]) > self.cs_params[i].upper) or \
                        (np.min(X[:, i]) < self.cs_params[i].lower):
                    raise RuntimeError('Some sample values from X are not in the given interval')
            elif isinstance(self.cs_params[i], CategoricalHyperparameter):
                unique_vals = set(X[:, i])
                if len(unique_vals) > len(self.cs_params[i].choices):
                    raise RuntimeError("There are some categoricals missing in the ConfigSpace specification for "
                                       "hyperparameter %s:" % self.cs_params[i].name)
            elif isinstance(self.cs_params[i], OrdinalHyperparameter):
                unique_vals = set(X[:, i])
                if len(unique_vals) > len(self.cs_params[i].sequence):
                    raise RuntimeError("There are some sequence-options missing in the ConfigSpace specification for "
                                       "hyperparameter %s:" % self.cs_params[i].name)
            elif isinstance(self.cs_params[i], Constant):
                # oddly, unparameterizedhyperparameter and constant are not supported. 
                # raise TypeError('Unsupported Hyperparameter: %s' % type(self.cs_params[i]))
                pass
                # unique_vals = set(X[:, i])
                # if len(unique_vals) > 1:
                #     raise RuntimeError('Got multiple values for Unparameterized (Constant) hyperparameter')
            else:
                raise TypeError('Unsupported Hyperparameter: %s' % type(self.cs_params[i]))

        if not np.issubdtype(X.dtype, np.float64):
            logging.warning('low level library expects X argument to be float')
        if not np.issubdtype(Y.dtype, np.float64):
            logging.warning('low level library expects Y argument to be float')

        # initialize all types as 0
        types = np.zeros(len(self.cs_params), dtype=np.uint)
        # retrieve the types and the bounds from the ConfigSpace 
        # TODO: Test if that actually works
        for i, hp in enumerate(self.cs_params):
            if isinstance(hp, CategoricalHyperparameter):
                types[i] = len(hp.choices)
                pcs[i] = (len(hp.choices), np.nan)
            elif isinstance(hp, OrdinalHyperparameter):
                types[i] = len(hp.sequence)
                pcs[i] = (len(hp.sequence), np.nan)
            elif isinstance(self.cs_params[i], NumericalHyperparameter):
                pcs[i] = (hp.lower, hp.upper)
            elif isinstance(self.cs_params[i], Constant):
                types[i] = 1
                pcs[i] = (1, np.nan)
            else:
                raise TypeError('Unsupported Hyperparameter: %s' % type(hp))

        # set forest options
        forest = reg.fanova_forest()
        forest.options.num_trees = n_trees
        forest.options.do_bootstrapping = bootstrapping
        forest.options.num_data_points_per_tree = X.shape[0] if points_per_tree is None else points_per_tree
        forest.options.tree_opts.max_features = (X.shape[1] * 7) // 10 if max_features is None else max_features

        forest.options.tree_opts.min_samples_to_split = min_samples_split
        forest.options.tree_opts.min_samples_in_leaf = min_samples_leaf
        forest.options.tree_opts.max_depth = max_depth
        forest.options.tree_opts.epsilon_purity = 1e-8

        # create data container and provide all the necessary information
        if seed is None:
            rng = reg.default_random_engine(np.random.randint(2 ** 31 - 1))
        else:
            rng = reg.default_random_engine(seed)
        data = reg.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(pcs):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for i in range(len(Y)):
            self.logger.debug("process datapoint: %s", str(X[i].tolist()))
            data.add_data_point(X[i].tolist(), Y[i])

        forest.fit(data, rng)

        self.the_forest = forest

        # initialize a dictionary with parameter dims
        self.variance_dict = dict()

        # getting split values
        forest_split_values = self.the_forest.all_split_values()

        # all midpoints and interval sizes treewise for the whole forest
        self.all_midpoints = []
        self.all_sizes = []

        # compute midpoints and interval sizes for variables in each tree
        for tree_split_values in forest_split_values:
            sizes = []
            midpoints = []
            for i, split_vals in enumerate(tree_split_values):
                if np.isnan(pcs[i][1]):  # categorical parameter
                    # check if the tree actually splits on this parameter
                    if len(split_vals) > 0:
                        midpoints.append(split_vals)
                        sizes.append(np.ones(len(split_vals)))
                    # if not, simply append 0 as the value with the number of categories as the size, that way this
                    # parameter will get 0 importance from this tree.
                    else:
                        midpoints.append((0,))
                        sizes.append((pcs[i][0],))
                else:
                    # add bounds to split values
                    sv = np.array([pcs[i][0]] + list(split_vals) + [pcs[i][1]])
                    # compute midpoints and sizes
                    midpoints.append((1 / 2) * (sv[1:] + sv[:-1]))
                    sizes.append(sv[1:] - sv[:-1])

            self.all_midpoints.append(midpoints)
            self.all_sizes.append(sizes)

        # capital V in the paper
        self.trees_total_variances = []
        # dict of lists where the keys are tuples of the dimensions
        # and the value list contains \hat{f}_U for the individual trees
        # reset all the variance fractions computed
        self.trees_variance_fractions = {}
        self.V_U_total = {}
        self.V_U_individual = {}

        self.cutoffs = cutoffs
        self.set_cutoffs(cutoffs)

    def set_cutoffs(self, cutoffs=(-np.inf, np.inf), quantile=None):
        """
        Setting the cutoffs to constrain the input space
        
        To properly do things like 'improvement over default' the
        fANOVA now supports cutoffs on the y values. These will exclude
        parts of the parameters space where the prediction is not within
        the provided cutoffs. This is is specialization of 
        "Generalized Functional ANOVA Diagnostics for High Dimensional
        Functions of Dependent Variables" by Hooker.
        """
        if not (quantile is None):
            percentile1 = self.percentiles[quantile[0]]
            percentile2 = self.percentiles[quantile[1]]
            self.the_forest.set_cutoffs(percentile1, percentile2)
        else:

            self.cutoffs = cutoffs
            self.the_forest.set_cutoffs(cutoffs[0], cutoffs[1])

        # reset all the variance fractions computed
        self.trees_variance_fractions = {}
        self.V_U_total = {}
        self.V_U_individual = {}

        # recompute the trees' total variance
        self.trees_total_variance = self.the_forest.get_trees_total_variances()

    def __compute_marginals(self, dimensions):
        """
        Returns the marginal of selected parameters
                
        Parameters
        ----------
        dimensions: tuple
            Contains the indices of ConfigSpace for the selected parameters (starts with 0)
        """
        dimensions = tuple(dimensions)

        # check if values has been previously computed
        if dimensions in self.V_U_individual:
            return

        # otherwise make sure all lower order marginals have been
        # computed, if not compute them
        for k in range(1, len(dimensions)):
            for sub_dims in it.combinations(dimensions, k):
                if sub_dims not in self.V_U_total:
                    self.__compute_marginals(sub_dims)

        # now all lower order terms have been computed
        self.V_U_individual[dimensions] = []
        self.V_U_total[dimensions] = []
        for tree_idx in range(len(self.all_midpoints)):
            # collect all the midpoints and corresponding sizes for that tree
            midpoints = [self.all_midpoints[tree_idx][dim] for dim in dimensions]
            sizes = [self.all_sizes[tree_idx][dim] for dim in dimensions]
            stat = pyrfr.util.weighted_running_stats()

            prod_midpoints = it.product(*midpoints)
            prod_sizes = it.product(*sizes)

            sample = np.full(self.n_dims, np.nan, dtype=np.float)

            # make prediction for all midpoints and weigh them by the corresponding size
            for i, (m, s) in enumerate(zip(prod_midpoints, prod_sizes)):
                sample[list(dimensions)] = list(m)
                ls = self.the_forest.marginal_prediction_stat_of_tree(tree_idx, sample.tolist())
                # self.logger.debug("%s, %s", (sample, ls.mean()))
                if not np.isnan(ls.mean()):
                    stat.push(ls.mean(), np.prod(np.array(s)) * ls.sum_of_weights())

            # line 10 in algorithm 2
            # note that V_U^2 can be computed by var(\hat a)^2 - \sum_{subU} var(f_subU)^2
            # which is why, \hat{f} is never computed in the code, but
            # appears in the pseudocode
            V_U_total = np.nan
            V_U_individual = np.nan

            if stat.sum_of_weights() > 0:
                V_U_total = stat.variance_population()
                V_U_individual = stat.variance_population()
                for k in range(1, len(dimensions)):
                    for sub_dims in it.combinations(dimensions, k):
                        V_U_individual -= self.V_U_individual[sub_dims][tree_idx]
                V_U_individual = np.clip(V_U_individual, 0, np.inf)

            self.V_U_individual[dimensions].append(V_U_individual)
            self.V_U_total[dimensions].append(V_U_total)

    def quantify_importance(self, dims):
        if type(dims[0]) == str:
            idx = []
            for i, param in enumerate(dims):
                idx.append(self.cs.get_idx_by_hyperparameter_name(param))
            dimensions = idx
        # make sure that all the V_U values are computed for each tree
        else:
            dimensions = dims

        self.__compute_marginals(dimensions)

        importance_dict = {}

        for k in range(1, len(dimensions) + 1):
            for sub_dims in it.combinations(dimensions, k):
                if type(dims[0]) == str:
                    dim_names = []
                    for j, dim in enumerate(sub_dims):
                        dim_names.append(self.cs.get_hyperparameter_by_idx(dim))
                    dim_names = tuple(dim_names)
                    importance_dict[dim_names] = {}
                else:
                    importance_dict[sub_dims] = {}
                # clean here to catch zero variance in a trees
                non_zero_idx = np.nonzero([self.trees_total_variance[t] for t in range(self.n_trees)])
                if len(non_zero_idx[0]) == 0:
                    raise RuntimeError('Encountered zero total variance in all trees.')

                fractions_total = np.array([self.V_U_total[sub_dims][t] / self.trees_total_variance[t]
                                            for t in non_zero_idx[0]])
                fractions_individual = np.array([self.V_U_individual[sub_dims][t] / self.trees_total_variance[t]
                                                 for t in non_zero_idx[0]])

                if type(dims[0]) == str:
                    importance_dict[dim_names]['individual importance'] = np.mean(fractions_individual)
                    importance_dict[dim_names]['total importance'] = np.mean(fractions_total)
                    importance_dict[dim_names]['individual std'] = np.std(fractions_individual)
                    importance_dict[dim_names]['total std'] = np.std(fractions_total)
                else:
                    importance_dict[sub_dims]['individual importance'] = np.mean(fractions_individual)
                    importance_dict[sub_dims]['total importance'] = np.mean(fractions_total)
                    importance_dict[sub_dims]['individual std'] = np.std(fractions_individual)
                    importance_dict[sub_dims]['total std'] = np.std(fractions_total)

        return importance_dict

    def marginal_mean_variance_for_values(self, dimlist, values_to_predict):
        """
        Returns the marginal of selected parameters for specific values
                
        Parameters
        ----------
        dimlist: list
                Contains the indices of ConfigSpace for the selected parameters 
                (starts with 0) 
        values_to_predict: list
                Contains the values to be predicted
              
        Returns
        -------
        tuple 
            marginal mean prediction and corresponding variance estimate
        """
        sample = np.full(self.n_dims, np.nan, dtype=np.float)
        for i in range(len(dimlist)):
            sample[dimlist[i]] = values_to_predict[i]

        return self.the_forest.marginal_mean_variance_prediction(sample)

    def get_most_important_pairwise_marginals(self, params=None, n=10):
        """
        Returns the n most important pairwise marginals from the whole ConfigSpace
            
        Parameters
        ----------
        params: list of strings or ints
            If specified, limit analysis to those parameters. If ints, interpreting as indices from ConfigurationSpace
        n: int
             The number of most relevant pairwise marginals that will be returned
          
        Returns
        -------
        list: 
             Contains the n most important pairwise marginals
        """
        self.tot_imp_dict = OrderedDict()
        pairwise_marginals = []
        if params is None:
            dimensions = range(self.n_dims)
        else:
            if type(params[0]) == str:
                idx = []
                for i, param in enumerate(params):
                    idx.append(self.cs.get_idx_by_hyperparameter_name(param))
                dimensions = idx

            else:
                dimensions = params
        # pairs = it.combinations(dimensions,2)
        pairs = [x for x in it.combinations(dimensions, 2)]
        if params:
            n = len(list(pairs))
        for combi in pairs:
            pairwise_marginal_performance = self.quantify_importance(combi)
            tot_imp = pairwise_marginal_performance[combi]['individual importance']
            combi_names = [self.cs_params[combi[0]].name, self.cs_params[combi[1]].name]
            pairwise_marginals.append((tot_imp, combi_names[0], combi_names[1]))

        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)

        for marginal, p1, p2 in pairwise_marginal_performance[:n]:
            self.tot_imp_dict[(p1, p2)] = marginal
        self._dict = True

        return self.tot_imp_dict

    def get_triple_marginals(self, params=None):
        """
        Returns the n most important pairwise marginals from the whole ConfigSpace
            
        Parameters
        ----------
        params: list
             The parameters
          
        Returns
        -------
        list: 
             Contains most important triple marginals
        """
        self.tot_imp_dict = OrderedDict()
        triple_marginals = []
        if len(params) < 3:
            raise RuntimeError(
                'Number of parameters have to be greater than %i. At least 3 parameters needed' % len(params))
        if type(params[0]) == str:
            idx = []
            for i, param in enumerate(params):
                idx.append(self.cs.get_idx_by_hyperparameter_name(param))
            dimensions = idx

        else:
            dimensions = params

        triplets = [x for x in it.combinations(dimensions, 3)]
        for combi in triplets:
            triple_marginal_performance = self.quantify_importance(combi)
            tot_imp = triple_marginal_performance[combi]['individual importance']
            combi_names = [self.cs_params[combi[0]].name, self.cs_params[combi[1]].name, self.cs_params[combi[2]].name]
            triple_marginals.append((tot_imp, combi_names[0], combi_names[1], combi_names[2]))

        triple_marginal_performance = sorted(triple_marginals, reverse=True)
        if params:
            triple_marginal_performance = triple_marginal_performance[:len(list(triplets))]

        for marginal, p1, p2, p3 in triple_marginal_performance:
            self.tot_imp_dict[(p1, p2, p3)] = marginal

        return self.tot_imp_dict
