# License: MIT

import logging
import typing

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import threading
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
try:
    from sklearn.ensemble.base import _partition_estimators
    old_sk_version = True
except ModuleNotFoundError:
    from sklearn.ensemble._base import _partition_estimators
    old_sk_version = False

from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.constants import N_TREES


def _collect_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        out.append(prediction)


class skRandomForestWithInstances(AbstractModel):

    """Random forest that takes instance features into account.

    implement based on sklearn.ensemble.RandomForestRegressor

    Attributes
    ----------
    n_points_per_tree : int
    rf : RandomForestRegressor
        Only available after training
    unlog_y: bool
    seed : int
    types : np.ndarray
    bounds : list
    rng : np.random.RandomState
    logger : logging.logger
    """

    def __init__(self, types: np.ndarray,
                 bounds: typing.List[typing.Tuple[float, float]],
                 log_y: bool=False,
                 num_trees: int=N_TREES,
                 do_bootstrapping: bool=True,
                 n_points_per_tree: int=-1,
                 ratio_features: float=5. / 6.,
                 min_samples_split: int=3,
                 min_samples_leaf: int=3,
                 max_depth: int=2**20,
                 eps_purity: float=1e-8,
                 max_num_nodes: int=2**20,
                 seed: int=42,
                 n_jobs: int=None,
                 **kwargs):
        """
        Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : list
            Specifies the bounds for continuous features.
        log_y: bool
            y values (passed to this RF) are expected to be log(y) transformed;
            this will be considered during predicting
        num_trees : int
            The number of trees in the random forest.
        do_bootstrapping : bool
            Turns on / off bootstrapping in the random forest.
        n_points_per_tree : int
            Number of points per tree. If <= 0 X.shape[0] will be used
            in _train(X, y) instead
        ratio_features : float
            The ratio of features that are considered for splitting.
        min_samples_split : int
            The minimum number of data points to perform a split.
        min_samples_leaf : int
            The minimum number of data points in a leaf.
        max_depth : int
            The maximum depth of a single tree.
        eps_purity : float
            The minimum difference between two target values to be considered
            different
        max_num_nodes : int
            The maxmimum total number of nodes in a tree
        seed : int
            The seed that is passed to the random_forest_run library.
        n_jobs : int, default=None
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.
        """
        super().__init__(types, bounds, **kwargs)

        self.logger = logging.getLogger(self.__module__ + "." +
                                        self.__class__.__name__)

        self.log_y = log_y
        if self.log_y:
            raise NotImplementedError
        self.rng = np.random.RandomState(seed)

        self.num_trees = num_trees
        self.do_bootstrapping = do_bootstrapping
        max_features = None if ratio_features > 1.0 else \
            int(max(1, types.shape[0] * ratio_features))
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.epsilon_purity = eps_purity
        self.max_num_nodes = max_num_nodes

        self.n_points_per_tree = n_points_per_tree
        self.n_jobs = n_jobs

        self.rf = None  # type: RandomForestRegressor

    def _train(self, X: np.ndarray, y: np.ndarray):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """

        self.X = X
        self.y = y.flatten()

        if self.n_points_per_tree <= 0:
            self.num_data_points_per_tree = self.X.shape[0]
        else:
            self.num_data_points_per_tree = self.n_points_per_tree
        if old_sk_version:
            self.rf = RandomForestRegressor(
                n_estimators=self.num_trees,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                # max_samples=self.num_data_points_per_tree,
                max_leaf_nodes=self.max_num_nodes,
                min_impurity_decrease=self.epsilon_purity,
                bootstrap=self.do_bootstrapping,
                n_jobs=self.n_jobs,
                random_state=self.rng,
            )
        else:
            self.rf = RandomForestRegressor(
                n_estimators=self.num_trees,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_samples=self.num_data_points_per_tree,
                max_leaf_nodes=self.max_num_nodes,
                min_impurity_decrease=self.epsilon_purity,
                bootstrap=self.do_bootstrapping,
                n_jobs=self.n_jobs,
                random_state=self.rng,
            )
        self.rf.fit(self.X, self.y)
        return self

    def predict_mean_var(self, X: np.ndarray):
        if old_sk_version:
            check_is_fitted(self.rf, 'estimators_')
        else:
            check_is_fitted(self.rf)
        # Check data
        if X.ndim == 1:
            X = X.reshape((1, -1))
        X = self.rf._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.rf.n_estimators, self.rf.n_jobs)

        # collect the output of every estimator
        all_y_preds = list()

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.rf.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_collect_prediction)(e.predict, X, all_y_preds, lock)
            for e in self.rf.estimators_)
        all_y_preds = np.asarray(all_y_preds, dtype=np.float64)

        m = np.mean(all_y_preds, axis=0)
        v = np.var(all_y_preds, axis=0)
        return m, v

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' % (self.types.shape[0], X.shape[1]))

        if self.log_y:
            raise NotImplementedError
        else:
            means, vars_ = self.predict_mean_var(X)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized_over_instances(self, X: np.ndarray):
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Note
        ----
        This method overwrites the same method of ~smac.epm.base_epm.AbstractEPM;
        the following method is random forest specific
        and follows the SMAC2 implementation;
        it requires no distribution assumption
        to marginalize the uncertainty estimates

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if self.log_y:
            raise NotImplementedError
        else:
            return super().predict_marginalized_over_instances(X)
