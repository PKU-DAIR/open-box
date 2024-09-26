import pytest
from unittest.mock import MagicMock, patch
from openbox.core.ea.saea_advisor import SAEAAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import Observation
from ConfigSpace import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor


'''
urrogate-assisted evolutionary algorithm, SAEA
'''

def test_saea_advisor(configspace_tiny):
    config_space = configspace_tiny
    advisor = SAEAAdvisor(config_space, population_size=4, num_objectives=1, ea=RegularizedEAAdvisor(config_space, subset_size=2, population_size=4, num_objectives=1))
    advisor_eic = SAEAAdvisor(config_space, population_size=4, num_constraints=1, ea=RegularizedEAAdvisor(config_space, subset_size=2, population_size=4, num_constraints=1))

    advisor_ehvic = SAEAAdvisor(config_space, population_size=4, num_objectives=2, num_constraints=1, ref_point=[1,1],  ea=RegularizedEAAdvisor(config_space, subset_size=2, population_size=4, num_objectives=2, num_constraints=1))

    advisor_ehvi = SAEAAdvisor(config_space, population_size=4, num_objectives=2, ref_point=[1,1], ea=RegularizedEAAdvisor(config_space, subset_size=2, population_size=4, num_objectives=2))

    '''
    mesmo不需要ref_point
    '''
    advisor_mesmo = SAEAAdvisor(config_space, population_size=4, num_objectives=2, ea=RegularizedEAAdvisor(config_space, subset_size=2, population_size=4, num_objectives=2))

    advisor_mesmoc = SAEAAdvisor(config_space, population_size=4, num_objectives=2, num_constraints=1, ea=RegularizedEAAdvisor(config_space, subset_size=2, num_objectives=2, population_size=4, num_constraints=1))

    advisor_parego = SAEAAdvisor(config_space, population_size=4, acq='parego', num_objectives=2, num_constraints=1, ea=RegularizedEAAdvisor(config_space, subset_size=2, num_objectives=2, population_size=4, num_constraints=1))
    
    assert advisor.config_space == config_space
    assert advisor.gen_multiplier == 50
    assert advisor.is_models_trained is False

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2]]
    for i in range(5):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i][:1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)
        if i == 0:
            advisor.sel()
        #第二轮清空一下cache，不然会无法循环了 
        if i == 1:
            advisor.cached_config = []
        assert advisor.is_models_trained is True

        suggestion1 = advisor_eic.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i][:1], constraints=[1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_eic.update_observation(observation1)
        if i == 0:
            advisor_eic.sel()

        suggestion1 = advisor_ehvic.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], constraints=[1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_ehvic.update_observation(observation1)
        if i == 0:
            advisor_ehvic.sel()

        suggestion1 = advisor_ehvi.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_ehvi.update_observation(observation1)
        if i == 0:
            advisor_ehvi.sel()
            
        suggestion1 = advisor_mesmoc.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], constraints=[1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_mesmoc.update_observation(observation1)
        if i == 0:
            advisor_mesmoc.sel()

        suggestion1 = advisor_mesmo.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_mesmo.update_observation(observation1)
        if i == 0:
            advisor_mesmo.sel()

        suggestion1 = advisor_parego.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], constraints=[1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_parego.update_observation(observation1)
        if i == 0:
            advisor_parego.sel()