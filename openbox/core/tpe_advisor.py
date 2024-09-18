# License: MIT

import traceback
import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm

from openbox import logger
from openbox.core.base_advisor import BaseAdvisor


class TPE_Advisor(BaseAdvisor):
    # TODO：Add warm start
    def __init__(
            self,
            config_space,
            min_points_in_model=None,
            top_n_percent=15,
            num_samples=64,
            random_fraction=1 / 3,
            bandwidth_factor=3,
            min_bandwidth=1e-3,
            task_id='OpenBox',
            output_dir='logs',
            random_state=None,
            logger_kwargs: dict = None,
    ):
        super().__init__(
            config_space=config_space,
            num_objectives=1,
            num_constraints=0,
            ref_point=None,
            output_dir=output_dir,
            task_id=task_id,
            random_state=random_state,
            logger_kwargs=logger_kwargs,
        )

        self.top_n_percent = top_n_percent
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

        if self.min_points_in_model < len(self.config_space.get_hyperparameters()) + 1:
            self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        hps = self.config_space.get_hyperparameters()

        self.kde_vartypes = ""
        self.vartypes = []

        for h in hps:
            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)

        # store precomputed probs for the categorical parameters
        self.cat_probs = []

        self.good_config_rankings = dict()
        self.kde_models = dict()

    def get_suggestion(self, history=None):
        if history is None:
            history = self.history

        # use default as first config
        num_config_evaluated = len(history)
        if num_config_evaluated == 0:
            return self.config_space.get_default_configuration()

        # fit
        self.fit_kde_models(history)

        # If no model is available, sample random config
        if len(self.kde_models.keys()) == 0 or self.rng.rand() < self.random_fraction:
            return self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]

        best = np.inf
        best_vector = None

        try:
            l = self.kde_models['good'].pdf
            g = self.kde_models['bad'].pdf

            minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

            kde_good = self.kde_models['good']
            kde_bad = self.kde_models['bad']

            for i in range(self.num_samples):
                idx = self.rng.randint(0, len(kde_good.data))
                datum = kde_good.data[idx]
                vector = []

                for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                    bw = max(bw, self.min_bandwidth)
                    if t == 0:
                        bw = self.bw_factor * bw
                        try:
                            vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                        except:
                            logger.warning(
                                "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s" % (
                                    datum, kde_good.bw, m))
                            logger.warning("data in the KDE:\n%s" % kde_good.data)
                    else:

                        if self.rng.rand() < (1 - bw):
                            vector.append(int(m))
                        else:
                            vector.append(self.rng.randint(t))
                val = minimize_me(vector)

                if not np.isfinite(val):
                    logger.warning('sampled vector: %s has EI value %s' % (vector, val))
                    logger.warning("data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
                    logger.warning("bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
                    logger.warning("l(x) = %s" % (l(vector)))
                    logger.warning("g(x) = %s" % (g(vector)))

                    # right now, this happens because a KDE does not contain all values for a categorical parameter
                    # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                    # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
                    if np.isfinite(l(vector)):
                        best_vector = vector
                        break

                if val < best:
                    best = val
                    best_vector = vector

            if best_vector is None:
                logger.debug(
                    "Sampling based optimization with %i samples failed -> using random configuration" % self.num_samples)
                config = self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]
            else:
                logger.debug(
                    'best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                for i, hp_value in enumerate(best_vector):
                    if isinstance(
                            self.config_space.get_hyperparameter(
                                self.config_space.get_hyperparameter_by_idx(i)
                            ),
                            ConfigSpace.hyperparameters.CategoricalHyperparameter
                    ):
                        best_vector[i] = int(np.rint(best_vector[i]))
                try:
                    config = ConfigSpace.Configuration(self.config_space, vector=best_vector)
                except Exception as e:
                    logger.warning(("=" * 50 + "\n") * 3 +
                                        "Error converting configuration:\n%s" % best_vector +
                                        "\n here is a traceback:" +
                                        traceback.format_exc())
                    raise e

        except:
            logger.warning(
                "Sampling based optimization with %i samples failed\n %s \nUsing random configuration" % (
                    self.num_samples, traceback.format_exc()))
            config = self.sample_random_configs(self.config_space, 1, excluded_configs=history.configurations)[0]

        return config

    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = self.rng.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = self.rng.rand()
                    else:
                        datum[nan_idx] = self.rng.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array

    def fit_kde_models(self, history):
        '''
        Called by self.get_suggestion()
        '''
        num_config_successful = history.get_success_count()
        if num_config_successful <= self.min_points_in_model - 1:
            logger.debug("Only %i run(s) available, need more than %s -> can't build model!" % (
                num_config_successful, self.min_points_in_model + 1))
            return

        train_configs = history.get_config_array(transform='scale')
        train_losses = history.get_objectives(transform='infeasible').reshape(-1)

        n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0]) // 100)
        # n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
        n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * train_configs.shape[0]) // 100)

        # Refit KDE for the current budget
        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad = self.impute_conditional_data(train_configs[idx[n_good:n_good + n_bad]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        # more expensive crossvalidation method
        # bw_estimation = 'cv_ls'

        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes,
                                                   bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes,
                                                    bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models = {
            'good': good_kde,
            'bad': bad_kde
        }

        # update probs for the categorical parameters for later sampling
        logger.debug(
            'done building a new model based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n' % (
                n_good, n_bad, np.min(train_losses)))
