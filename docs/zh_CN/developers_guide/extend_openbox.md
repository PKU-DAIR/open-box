# Extend Algorithms in OpenBox

This guide will teach you how to extend algorithms in OpenBox.

You can implement a new surrogate model, a new acquisition function 
or a new acquisition function maximizer for Bayesian Optimization Advisor,
or implement an advisor with a totally new algorithm.

## Extend Bayesian Optimization Advisor

### Workflow of Bayesian Optimization Advisor

Let's start with understanding the workflow of Bayesian optimization Advisor.

In each iteration, `Advisor.get_suggestion()` is called to generate a new config.
Here are the main steps in `get_suggestion()`:
```python
def get_suggestion(self, ...):
    self.surrogate_model.train(...)
    self.acquisition_function.update(...)
    challengers = self.acq_optimizer.maximize(...)
    ...
```
First, the surrogate model is trained with the latest data.

Second, the acquisition function is updated with the surrogate model and additional information.

Third, the acquisition function maximizer samples points to maximize the acquisition function.

Please refer to `openbox/core/generic_advisor.py` for more details.

### Implement a New Surrogate Model

To implement a new surrogate model, inherit the `AbstractModel` class.
The methods `_train()` and `_predict()` should be implemented.

For `_train()`, the surrogate model should be updated with the latest data.

For `_predict()`, a new batch of data is given, and the method should return 
predicted mean and variance of the batch.

Please refer to `openbox/surrogate/base/base_model.py` for more details.

Here is an example that implements a surrogate using RandomForest in scikit-learn:

```python
import typing

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import threading
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._base import _partition_estimators

from openbox.surrogate.base.base_model import AbstractModel


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
    def __init__(self, types: np.ndarray,
                 bounds: typing.List[typing.Tuple[float, float]],
                 num_trees: int=10,
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
        X : np.ndarray [n_samples, n_features]
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

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features]

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

        means = np.mean(all_y_preds, axis=0)
        vars_ = np.var(all_y_preds, axis=0)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))
```

### Implement a New Acquisition Function

To implement a new acquisition function, inherit the `AbstractAcquisitionFunction` class.
The method `_compute()` should be implemented, which calculates the acquisition function 
values of a new batch of points. 
Call `self.model.predict_marginalized_over_instances(X)` to get predicted mean and variance 
of the points from the surrogate model.

Please refer to `openbox/acquisition_function/acquisition.py` for more details.

Here is an example of the probability of improvement (PI) acquisition function:

```python
import numpy as np
from scipy.stats import norm
from openbox.acquisition_function import AbstractAcquisitionFunction

class PI(AbstractAcquisitionFunction):
    def __init__(self,
                 model,
                 par: float = 0.0,
                 **kwargs):

        r"""Computes the probability of improvement for a given x over the best so far value as
        acquisition value.

        :math:`P(f_{t+1}(\mathbf{X})\geq f(\mathbf{X^+})) :=
        \Phi(\frac{\mu(\mathbf{X}) - f(\mathbf{X^+})}{\sigma(\mathbf{X})})`,
        with :math:`f(X^+)` as the incumbent and :math:`\Phi` the cdf of the standard normal

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(PI, self).__init__(model)
        self.long_name = 'Probability of Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the PI value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate PI. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N, 1)
            PI of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<float>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        return norm.cdf((self.eta - m - self.par) / std)
```

### Implement a New Acquisition Function Maximizer

To implement a new acquisition function maximizer, inherit the `AcquisitionFunctionMaximizer` class.
The method `_maximize()` should be implemented, which returns an iterable of tuples, 
consisting of the acquisition function value and the configuration.

Please refer to `openbox/acq_optimizer/basic_maximizer.py` for more details.

### Modify Advisor with Your Own Components

To use your own surrogate model / acquisition function / acquisition function maximizer, 
simply replace the attribute in `Advisor` as following:

```python
from openbox import Advisor
advisor = Advisor(...)
advisor.surrogate_model = MySurrogateModel(...)
advisor.acquisition_function = MyAcquisitionFunction(...)
advisor.acq_optimizer = MyAcquisitionFunctionMaximizer(...)
```

This is the quickest way to use custom components.
To integrate new algorithms into Bayesian optimization `Advisor`,
please modify the initialization process of `Advisor`.

## Implement a New Advisor

The advisor is designed to be a config generator.
In each iteration, `advisor.get_suggestion()` is called to generate a new config.
After evaluation, the `Observation` is updated into advisor by `advisor.update_observation()`.

To implement a new advisor, inherit the `BaseAdvisor` class.
In most cases, implement the `get_suggestion()` method is sufficient. 
This method should return a new config to be evaluated.

To do extra actions during updating history data, please modify `update_observation()` in advisor.

To suggest multiple configs at a time in parallel settings, implement `get_suggestions()` instead.

Here is an example of a naive random advisor without duplication check:

```python
from openbox.core.base_advisor import BaseAdvisor

class RandomAdvisor(BaseAdvisor):
    def get_suggestion(self):
        config = self.config_space.sample_configuration()
        return config
```

And here is an example multi-objective EHVI advisor implemented using [BoTorch](https://github.com/pytorch/botorch):
```python
import warnings
from typing import List
import numpy as np
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter, Configuration

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler

from openbox import Observation, logger
from openbox.core.base_advisor import BaseAdvisor
from openbox.utils.config_space.space_utils import get_config_from_dict


class BoTorchEHVIAdvisor(BaseAdvisor):
    """
    An Advisor using BoTorch's qEHVI acquisition function.

    Caution: BoTorch maximizes the objectives.
    """
    def __init__(
            self,
            config_space,
            num_objectives: int,
            ref_point,
            init_num=-1,
            NUM_RESTARTS=10,
            RAW_SAMPLES=512,
            MC_SAMPLES=128,
            output_dir='logs',
            task_id='BoTorchEHVI',
            seed=None,
            logger_kwargs: dict = None,
    ):
        assert num_objectives > 1
        for hp in config_space.get_hyperparameters():
            assert isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter))
        assert ref_point is not None

        super().__init__(
            config_space=config_space,
            num_objectives=num_objectives,
            num_constraints=0,
            ref_point=ref_point,
            output_dir=output_dir,
            task_id=task_id,
            random_state=seed,
            logger_kwargs=logger_kwargs,
        )

        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        self.NUM_RESTARTS = NUM_RESTARTS
        self.RAW_SAMPLES = RAW_SAMPLES
        self.MC_SAMPLES = MC_SAMPLES

        self.problem_dim = len(config_space.get_hyperparameters())
        self.standard_bounds = torch.zeros(2, self.problem_dim, **self.tkwargs)
        self.standard_bounds[1] = 1
        self.problem_bounds = self.get_problem_bounds()

        # Caution: BoTorch maximizes the objectives
        self.botorch_ref_point = -1.0 * torch.tensor(ref_point, **self.tkwargs)

        # initial design
        self.init_num = init_num if init_num > 0 else 2 * (self.problem_dim + 1)
        logger.info(f'init_num: {self.init_num}')
        self.init_configs = self.generate_initial_configs(self.init_num)

    def get_suggestion(self):
        n_history = len(self.history)
        if n_history < self.init_num:
            logger.info(f'Initial iter {n_history + 1}/{self.init_num}')
            return self.init_configs[n_history]

        X = self.history.get_config_array(transform='numerical')
        Y = self.history.get_objectives(transform='failed')

        train_x = torch.tensor(X, **self.tkwargs)  # train_x is not normalized
        train_obj = -1.0 * torch.tensor(Y, **self.tkwargs)  # Caution: BoTorch maximizes the objectives

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll, model = self.initialize_model(train_x, train_obj)

        # fit the models
        fit_gpytorch_mll(mll)

        # define the acquisition modules using a QMC sampler
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.MC_SAMPLES]))

        # optimize acquisition functions and get new candidate
        new_x = self.optimize_qehvi(
            model, train_x, train_obj, qehvi_sampler
        )
        config = self.tensor2configs(new_x)[0]
        logger.info(f'Get suggestion. new_x: {new_x}, config: {config}')
        return config

    def update_observation(self, observation: Observation):
        logger.info(f'Update observation: {observation}')
        return super().update_observation(observation)

    def generate_initial_configs(self, init_num):
        x = draw_sobol_samples(bounds=self.problem_bounds, n=init_num, q=1).squeeze(1)
        configs = self.tensor2configs(x)
        return configs

    def initialize_model(self, train_x, train_obj):
        # define models for objective
        train_x = normalize(train_x, self.problem_bounds)
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i: i + 1]
            model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            models.append(model)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def optimize_qehvi(self, model, train_x, train_obj, sampler):
        """Optimizes the qEHVI acquisition function, and returns a new candidate."""
        # partition non-dominated space into disjoint rectangles
        with torch.no_grad():
            pred = model.posterior(normalize(train_x, self.problem_bounds)).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=self.botorch_ref_point,
            Y=pred,
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.botorch_ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.standard_bounds,
            q=1,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        # observe new values
        new_x = unnormalize(candidates.detach(), bounds=self.problem_bounds)
        return new_x

    def get_problem_bounds(self):
        bounds = []
        for hp in self.config_space.get_hyperparameters():
            bounds.append((hp.lower, hp.upper))
        bounds = torch.tensor(bounds, **self.tkwargs).transpose(0, 1)
        return bounds

    def tensor2configs(self, x: torch.Tensor) -> List[Configuration]:
        """
        Convert x (torch tensor) to a list of Configurations
        x should be unnormalized.
        """
        assert x.dim() == 2, f'Expected 2-d tensor, got {x.dim()}'
        configs = []
        for i in range(x.shape[0]):
            config_dict = dict()
            for j, hp in enumerate(self.config_space.get_hyperparameters()):
                config_dict[hp.name] = x[i, j].item()
            config = get_config_from_dict(self.config_space, config_dict)
            configs.append(config)
        return configs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from openbox.benchmark.objective_functions.synthetic import BraninCurrin

    problem = BraninCurrin()
    advisor = BoTorchEHVIAdvisor(
        config_space=problem.config_space,
        num_objectives=problem.num_objectives,
        ref_point=problem.ref_point,
        init_num=-1,
        output_dir='logs',
        task_id='BoTorchEHVI',
        seed=1234,
    )

    n_iter = 20
    for i in range(n_iter):
        logger.info(f'=== Iteration {i + 1}/{n_iter} ===')
        config = advisor.get_suggestion()
        result = problem(config)
        observation = Observation(config=config, objectives=result['objectives'])
        advisor.update_observation(observation)

    history = advisor.get_history()
    history.plot_hypervolumes(optimal_hypervolume=problem.max_hv, logy=True)
    plt.show()
```
