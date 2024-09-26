import pytest

from openbox.acquisition_function.acquisition import EI
from openbox.acquisition_function.multi_objective_acquisition import USeMO


@pytest.fixture
def acq_func_ei(surrogate_model_gp, history_single_obs):
    surrogate_model = surrogate_model_gp
    ei = EI(surrogate_model)

    X = history_single_obs.get_config_array(transform='scale')
    Y = history_single_obs.get_objectives(transform='infeasible')

    surrogate_model.train(X, Y)

    ei.update(model=surrogate_model,
              constraint_models=None,
              eta=history_single_obs.get_incumbent_value(),
              num_data=len(history_single_obs)
              )

    return ei


@pytest.fixture
def acq_func_usemo(configspace_tiny, surrogate_model_gp, history_double_obs):
    surrogate_model1 = surrogate_model_gp
    surrogate_model2 = surrogate_model_gp
    usemo = USeMO([surrogate_model1, surrogate_model2], config_space=configspace_tiny)

    X = history_double_obs.get_config_array(transform='scale')
    Y = history_double_obs.get_objectives(transform='infeasible')
    cY = history_double_obs.get_constraints(transform='bilog')

    surrogate_model1.train(X, Y[:, 0])
    surrogate_model1.train(X, Y[:, 1])

    usemo.update(model=[surrogate_model1, surrogate_model2],
                 constraint_models=None,
                 constraint_perfs=cY,  # for MESMOC
                 eta=history_double_obs.get_mo_incumbent_values(),
                 num_data=len(history_double_obs),
                 X=X, Y=Y)

    return usemo
