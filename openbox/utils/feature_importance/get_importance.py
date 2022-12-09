import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox import logger


def get_fanova_importance(X: np.ndarray, Y: np.ndarray, config_space: ConfigurationSpace, **kwargs):
    """
    Get feature importance using fANOVA.

    Parameters
    ----------
    X : np.ndarray
        Feature (hyperparameter) values. Shape: (n_samples, n_features).
    Y : np.ndarray
        Objective values. Shape: (n_samples,).
    config_space : ConfigurationSpace
        Configuration space.
    **kwargs
        Keyword arguments for fANOVA.

    Returns
    -------
    feature_importance : np.ndarray
        Feature importance. Shape: (n_features,).
    """
    try:
        import pyrfr.regression as reg
        import pyrfr.util
    except ModuleNotFoundError:
        logger.error(
            'To use fANOVA feature importance analysis, please install pyrfr: '
            'https://open-box.readthedocs.io/en/latest/installation/install_pyrfr.html'
        )
        raise
    from openbox.utils.feature_importance.fanova import fANOVA

    assert X.ndim == 2 and X.shape[0] > 0
    assert Y.ndim == 1 and Y.shape[0] == X.shape[0]

    # create an instance of fanova with data for the random forest and the configSpace
    f = fANOVA(X=X, Y=Y, config_space=config_space, seed=1, **kwargs)

    # marginal for individual parameter
    feature_importance = []
    for param in config_space.get_hyperparameter_names():
        p_list = (param,)
        res = f.quantify_importance(p_list)
        individual_importance = res[(param,)]['individual importance']
        feature_importance.append(individual_importance)
    feature_importance = np.array(feature_importance)
    return feature_importance


def get_shap_importance(X: np.ndarray, Y: np.ndarray, **kwargs):
    """
    Get feature importance using SHAP.

    Parameters
    ----------
    X : np.ndarray
        Feature (hyperparameter) values. Shape: (n_samples, n_features).
    Y : np.ndarray
        Objective values. Shape: (n_samples,).
    **kwargs
        Keyword arguments for LightGBM.

    Returns
    -------
    feature_importance : np.ndarray
        Feature importance. Shape: (n_features,).
    shap_values : np.ndarray
        SHAP values. Shape: (n_samples, n_features).
    """
    try:
        import shap
        from lightgbm import LGBMRegressor
    except ModuleNotFoundError:
        logger.error(
            'Please install shap and lightgbm to use SHAP feature importance analysis. '
            'Run "pip install shap lightgbm"!'
        )
        raise

    assert X.ndim == 2 and X.shape[0] > 0
    assert Y.ndim == 1 and Y.shape[0] == X.shape[0]

    # Fit a LightGBMRegressor with observations
    lgbr = LGBMRegressor(n_jobs=1, random_state=1, **kwargs)
    lgbr.fit(X, Y)
    explainer = shap.TreeExplainer(lgbr)
    shap_values = explainer.shap_values(X)
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    return feature_importance, shap_values
