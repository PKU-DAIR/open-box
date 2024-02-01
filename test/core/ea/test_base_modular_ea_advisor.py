import pytest
from unittest.mock import MagicMock, patch
from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor, Individual
from ConfigSpace import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_modular_ea_advisor_initialization(configspace_tiny):
    config_space = configspace_tiny
    advisor = ModularEAAdvisor(config_space, population_size=4, num_objectives=2)
    assert advisor.config_space == config_space
    assert advisor.required_evaluation_count == advisor.population_size
    assert advisor.filled_up is False
    assert advisor.cached_config == []
    assert advisor.uneval_config == []
    assert advisor.next_population == []
    assert advisor.auto_step is True
    assert advisor.strict_auto_step is True
    assert advisor.skip_gen_population is False
    assert advisor.filter_gen_population is None
    assert advisor.keep_unexpected_population is True
    assert advisor.save_cached_configuration is True

    with pytest.raises(NotImplementedError):
        advisor.get_suggestion()

