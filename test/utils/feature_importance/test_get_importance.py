from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.feature_importance.get_importance import get_fanova_importance, get_shap_importance
import numpy as np


def test_get_importance(configspace_tiny, func_brain):
    cs = configspace_tiny
    func = func_brain

    configs = cs.sample_configuration(10)
    perfs = np.zeros(10)

    for i in range(perfs.shape[0]):
        perfs[i] = func(configs[i])['objectives'][0]

    configs = convert_configurations_to_array(configs)

    importances_shap = get_shap_importance(configs, perfs)
    assert len(importances_shap) == 2

    importances_fanova = get_fanova_importance(configs, perfs, cs)
    assert len(importances_fanova) == 2
