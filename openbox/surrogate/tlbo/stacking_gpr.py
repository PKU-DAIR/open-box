# License: MIT

import numpy as np
import time
from openbox import logger, History
from openbox.surrogate.tlbo.base import BaseTLSurrogate
from openbox.core.base import build_surrogate

_scale_method = 'scale'


class SGPR(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed,
                 surrogate_type='rf', num_src_hpo_trial=50):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'sgpr'

        self.alpha = 0.95

        self.base_regressors = list()
        self.num_configs = list()
        self.final_regressor = None
        self.final_num = 0

        self.iteration_id = 0
        self.index_mapper = dict()
        self.get_regressor(normalize=_scale_method)

    def get_regressor(self, normalize):
        # Train transfer learning regressor.

        if self.source_hpo_data is None:
            logger.warning('No history BO data provided, resort to naive BO optimizer without TL.')
            return

        assert isinstance(self.source_hpo_data, list)

        start_time = time.time()
        self.source_surrogates = list()
        for idx, task_history in enumerate(self.source_hpo_data):
            assert isinstance(task_history, History)

            logger.info('Building the %d-th residual GPs.' % idx)

            X = task_history.get_config_array(transform=normalize)[:self.num_src_hpo_trial]
            y = task_history.get_objectives(transform='infeasible')[:self.num_src_hpo_trial]
            y = y.reshape(-1)  # single objective

            self.train_regressor(X, y)
        logger.info('Building the source surrogate took %.3fs.' % (time.time() - start_time))

    def train_regressor(self, X, y, is_top=False):
        model = build_surrogate(self.surrogate_type, self.config_space,
                                np.random.RandomState(self.random_seed))
        if is_top:
            self.final_num = len(X)
        else:
            self.num_configs.append(len(X))

        if len(self.base_regressors) == 0 or is_top:
            model.train(X, y)
        else:
            stacked_mu, _ = self.calculate_stacked_results(X)
            stacked_mu = np.reshape(stacked_mu, y.shape)

            model.train(X, y - stacked_mu)

        if not is_top:
            self.base_regressors.append(model)
        else:
            self.final_regressor = model

    def train(self, X: np.ndarray, y: np.array):
        # Train the final regressor.
        self.train_regressor(X, y, is_top=True)
        self.iteration_id += 1

    def calculate_stacked_results(self, X: np.ndarray, include_top=False):
        stacked_mu, stacked_sigma = np.zeros(len(X)), np.ones(len(X))
        for i, model in enumerate(self.base_regressors):
            mu, sigma = model.predict(X)
            mu, sigma = mu.flatten(), sigma.flatten()

            prior_size = 0 if i == 0 else self.num_configs[i - 1]
            cur_size = self.num_configs[i]
            beta = self.alpha * cur_size / (self.alpha * cur_size + prior_size)

            stacked_mu += mu
            stacked_sigma = np.power(sigma, beta) * np.power(stacked_sigma, 1 - beta)

        if include_top:
            mu, sigma = self.final_regressor.predict(X)
            mu, sigma = mu.flatten(), sigma.flatten()

            prior_size = self.num_configs[-1]
            cur_size = self.final_num
            beta = self.alpha * cur_size / (self.alpha * cur_size + prior_size)

            stacked_mu += mu
            stacked_sigma = np.power(sigma, beta) * np.power(stacked_sigma, 1 - beta)

        return stacked_mu, stacked_sigma

    def predict(self, X: np.array):
        mu, sigma = self.calculate_stacked_results(X, include_top=True)
        return np.array(mu).reshape(-1, 1), np.array(sigma).reshape(-1, 1)
