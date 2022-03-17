# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

import numpy as np
from openbox.surrogate.base.gp import GaussianProcess
from openbox.surrogate.base.gp_mcmc import GaussianProcessMCMC
from openbox.surrogate.base.gp_base_prior import HorseshoePrior, LognormalPrior
from openbox.surrogate.base.gp_kernels import ConstantKernel, Matern, HammingKernel, WhiteKernel, RBF


def create_gp_model(model_type, config_space, types, bounds, rng):
    """
        Construct the Gaussian process surrogate that is capable of dealing with categorical hyperparameters.
    """
    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
    )

    cont_dims = np.nonzero(types == 0)[0].astype(np.int)
    cat_dims = np.nonzero(types != 0)[0].astype(np.int)

    if len(cont_dims) > 0:
        exp_kernel = Matern(
            np.ones([len(cont_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
            nu=2.5,
            operate_on=cont_dims,
        )

    if len(cat_dims) > 0:
        ham_kernel = HammingKernel(
            np.ones([len(cat_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
            operate_on=cat_dims,
        )

    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )

    if len(cont_dims) > 0 and len(cat_dims) > 0:
        # both
        kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
    elif len(cont_dims) > 0 and len(cat_dims) == 0:
        # only cont
        kernel = cov_amp * exp_kernel + noise_kernel
    elif len(cont_dims) == 0 and len(cat_dims) > 0:
        # only cont
        kernel = cov_amp * ham_kernel + noise_kernel
    else:
        raise ValueError()

    # seed = rng.randint(0, 2 ** 20)
    if model_type == 'gp_mcmc':
        n_mcmc_walkers = 3 * len(kernel.theta)
        if n_mcmc_walkers % 2 == 1:
            n_mcmc_walkers += 1
        model = GaussianProcessMCMC(
            configspace=config_space,
            types=types,
            bounds=bounds,
            kernel=kernel,
            n_mcmc_walkers=n_mcmc_walkers,
            chain_length=250,
            burnin_steps=250,
            normalize_y=True,
            seed=rng.randint(low=0, high=10000),
        )
    elif model_type == 'gp':
        model = GaussianProcess(
            configspace=config_space,
            types=types,
            bounds=bounds,
            kernel=kernel,
            normalize_y=True,
            seed=rng.randint(low=0, high=10000),
        )
    elif model_type == 'gp_rbf':
        rbf_kernel = RBF(
            length_scale=1,
            length_scale_bounds=(1e-3, 1e2),
        )
        model = GaussianProcess(
            configspace=config_space,
            types=types,
            bounds=bounds,
            alpha=1e-10,    # Fix RBF kernel error  # todo: might conflict with restart training
            kernel=rbf_kernel,  # todo: add white kernel and check MESMO
            normalize_y=False,  # todo confirm
            seed=rng.randint(low=0, high=10000),
        )
    else:
        raise ValueError("Invalid surrogate str %s!" % model_type)
    return model
