import pytest

from openbox.acquisition_function.acquisition import EI
from openbox.acquisition_function.multi_objective_acquisition import USeMO,MESMO,MESMOC,MESMOC2
from openbox.acquisition_function.mc_multi_objective_acquisition import MCEHVI,MCParEGO,MCParEGOC,MCEHVIC
from openbox.utils.multi_objective import NondominatedPartitioning
from openbox.acquisition_function.mc_acquisition import MCEI,MCEIC


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
def acq_func_mcei(configspace_tiny, surrogate_model_gp, history_single_obs):
    surrogate_model = surrogate_model_gp
    mcei = MCEI(surrogate_model)

    X = history_single_obs.get_config_array(transform='scale')
    Y = history_single_obs.get_objectives(transform='infeasible')

    surrogate_model.train(X, Y)

    mcei.update(model=surrogate_model,
              constraint_models=None,
              eta=history_single_obs.get_incumbent_value(),
              num_data=len(history_single_obs)
              )
    mcei([configspace_tiny.sample_configuration()])
    return mcei


@pytest.fixture
def acq_func_mceic(configspace_tiny, surrogate_model_gp, history_single_obs):
    surrogate_model = surrogate_model_gp
    constriant_model = surrogate_model_gp
    mceic = MCEIC(surrogate_model, [constriant_model])

    X = history_single_obs.get_config_array(transform='scale')
    Y = history_single_obs.get_objectives(transform='infeasible')
    cY = history_single_obs.get_constraints(transform='bilog')

    surrogate_model.train(X, Y)

    mceic.update(model=surrogate_model,
              constraint_models=[constriant_model],
              constraint_perfs=cY,
              eta=history_single_obs.get_incumbent_value(),
              num_data=len(history_single_obs)
              )
    mceic([configspace_tiny.sample_configuration()])
    return mceic


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
    usemo([configspace_tiny.sample_configuration()])

    return usemo

@pytest.fixture
def acq_func_mesmo(configspace_tiny, surrogate_model_gp_rbf, history_double_obs):
    surrogate_model1 = surrogate_model_gp_rbf
    surrogate_model2 = surrogate_model_gp_rbf
    memso = MESMO([surrogate_model1, surrogate_model2], config_space=configspace_tiny)

    X = history_double_obs.get_config_array(transform='scale')
    Y = history_double_obs.get_objectives(transform='infeasible')
    cY = history_double_obs.get_constraints(transform='bilog')

    surrogate_model1.train(X, Y[:, 0])
    surrogate_model1.train(X, Y[:, 1])

    memso.update(model=[surrogate_model1, surrogate_model2],
                 constraint_models=None,
                 constraint_perfs=cY,  # for MESMOC
                 eta=history_double_obs.get_mo_incumbent_values(),
                 num_data=len(history_double_obs),
                 X=X, Y=Y)
    memso([configspace_tiny.sample_configuration()])
    return memso


@pytest.fixture
def acq_func_mesmoc(configspace_tiny, surrogate_model_gp_rbf, history_double_cons_obs):
    surrogate_model1 = surrogate_model_gp_rbf
    surrogate_model2 = surrogate_model_gp_rbf
    surrogate_model3 = surrogate_model_gp_rbf
    surrogate_model4 = surrogate_model_gp_rbf
    memsoc = MESMOC([surrogate_model1, surrogate_model2], [surrogate_model3, surrogate_model4], config_space=configspace_tiny)

    X = history_double_cons_obs.get_config_array(transform='scale')
    Y = history_double_cons_obs.get_objectives(transform='infeasible')
    cY = history_double_cons_obs.get_constraints(transform='bilog')

    surrogate_model1.train(X, Y[:, 0])
    surrogate_model1.train(X, Y[:, 1])

    memsoc.update(model=[surrogate_model1, surrogate_model2],
                 constraint_models= [surrogate_model3, surrogate_model4],
                 constraint_perfs=cY,  # for MESMOC
                 eta=history_double_cons_obs.get_mo_incumbent_values(),
                 num_data=len(history_double_cons_obs),
                 X=X, Y=Y)
    memsoc([configspace_tiny.sample_configuration()])
    return memsoc


@pytest.fixture
def acq_func_mesmoc2(configspace_tiny, surrogate_model_gp_rbf, history_double_cons_obs):
    surrogate_model1 = surrogate_model_gp_rbf
    surrogate_model2 = surrogate_model_gp_rbf
    surrogate_model3 = surrogate_model_gp_rbf
    surrogate_model4 = surrogate_model_gp_rbf
    memsoc2 = MESMOC2([surrogate_model1, surrogate_model2], [surrogate_model3, surrogate_model4], config_space=configspace_tiny)

    X = history_double_cons_obs.get_config_array(transform='scale')
    Y = history_double_cons_obs.get_objectives(transform='infeasible')
    cY = history_double_cons_obs.get_constraints(transform='bilog')

    surrogate_model1.train(X, Y[:, 0])
    surrogate_model1.train(X, Y[:, 1])

    memsoc2.update(model=[surrogate_model1, surrogate_model2],
                 constraint_models= [surrogate_model3, surrogate_model4],
                 constraint_perfs=cY,  # for MESMOC
                 eta=history_double_cons_obs.get_mo_incumbent_values(),
                 num_data=len(history_double_cons_obs),
                 X=X, Y=Y)
    #计算这个点的mesmoc值,但是这里只是为了测试，严格来说要勇敢X中的config，X含有4个config，最后后一次性计算他们的acq
    memsoc2([configspace_tiny.sample_configuration()])
    return memsoc2
'''
cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
self.acquisition_function.update(model=self.surrogate_model,
                                    constraint_models=self.constraint_models,
                                    cell_lower_bounds=cell_bounds[0],
                                    cell_upper_bounds=cell_bounds[1])
要主动传入cell_bound，update，并且update这些属性
'''
@pytest.fixture
def acq_func_mc_ehvic(configspace_tiny, surrogate_model_gp_rbf, history_double_cons_obs):
    surrogate_model1 = surrogate_model_gp_rbf
    surrogate_model2 = surrogate_model_gp_rbf
    surrogate_model3 = surrogate_model_gp_rbf
    surrogate_model4 = surrogate_model_gp_rbf
    mc_ehvic = MCEHVIC([surrogate_model1, surrogate_model2], [surrogate_model3, surrogate_model4], config_space=configspace_tiny, ref_point=[1,1])

    X = history_double_cons_obs.get_config_array(transform='scale')
    Y = history_double_cons_obs.get_objectives(transform='infeasible')
    cY = history_double_cons_obs.get_constraints(transform='bilog')
    partitioning = NondominatedPartitioning(2, Y)
    surrogate_model1.train(X, Y[:, 0])
    surrogate_model1.train(X, Y[:, 1])
    cell_bounds = partitioning.get_hypercell_bounds(ref_point=[1,1])
    mc_ehvic.update(model=[surrogate_model1, surrogate_model2],
                 constraint_models= [surrogate_model3, surrogate_model4],
                 constraint_perfs=cY,  # for MESMOC
                 eta=history_double_cons_obs.get_mo_incumbent_values(),
                 num_data=len(history_double_cons_obs),
                 X=X, Y=Y,
                cell_lower_bounds=cell_bounds[0],cell_upper_bounds=cell_bounds[1])
    #计算这个点的mesmoc值,但是这里只是为了测试，严格来说要勇敢X中的config，X含有4个config，最后后一次性计算他们的acq
    mc_ehvic([configspace_tiny.sample_configuration()])
    return mc_ehvic




@pytest.fixture
def acq_func_mc_parEGOC(configspace_tiny, surrogate_model_gp_rbf, history_double_cons_obs):
    surrogate_model1 = surrogate_model_gp_rbf
    surrogate_model2 = surrogate_model_gp_rbf
    surrogate_model3 = surrogate_model_gp_rbf
    surrogate_model4 = surrogate_model_gp_rbf
    mc_parEGOC = MCParEGOC([surrogate_model1, surrogate_model2], [surrogate_model3, surrogate_model4], config_space=configspace_tiny, ref_point=[1,1])

    X = history_double_cons_obs.get_config_array(transform='scale')
    Y = history_double_cons_obs.get_objectives(transform='infeasible')
    cY = history_double_cons_obs.get_constraints(transform='bilog')
    partitioning = NondominatedPartitioning(2, Y)
    surrogate_model1.train(X, Y[:, 0])
    surrogate_model1.train(X, Y[:, 1])
    cell_bounds = partitioning.get_hypercell_bounds(ref_point=[1,1])
    mc_parEGOC.update(model=[surrogate_model1, surrogate_model2],
                 constraint_models= [surrogate_model3, surrogate_model4],
                 constraint_perfs=cY,  # for MESMOC
                 eta=history_double_cons_obs.get_mo_incumbent_values(),
                 num_data=len(history_double_cons_obs),
                 X=X, Y=Y,
                cell_lower_bounds=cell_bounds[0],cell_upper_bounds=cell_bounds[1])
    #计算这个点的mesmoc值,但是这里只是为了测试，严格来说要勇敢X中的config，X含有4个config，最后后一次性计算他们的acq
    mc_parEGOC([configspace_tiny.sample_configuration()])
    return mc_parEGOC





