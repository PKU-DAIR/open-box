import pytest
from openbox import space as sp
from openbox.utils.history import MultiStartHistory, History, Observation
from openbox.utils.constants import SUCCESS, FAILED
from openbox.utils.config_space import ConfigurationSpace


@pytest.fixture
def history_single_single_obs():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(sp.Real("x1", 0, 1, default_value=0))
    # Create a History instance
    config1 = config_space.sample_configuration()
    config2 = config_space.sample_configuration()
    config3 = config_space.sample_configuration()

    # Set up observations with different objective values
    observation1 = Observation(config=config1, objectives=[0.1], trial_state=SUCCESS)
    observation2 = Observation(config=config2, objectives=[0.3], trial_state=SUCCESS)
    observation3 = Observation(config=config3, objectives=[0.5], trial_state=SUCCESS)

    history = History(
        num_objectives=1,
        config_space=config_space
    )

    # Add observations to history
    history.update_observations([observation1, observation2, observation3])

    return history

@pytest.fixture
def history_single(configspace_tiny):
    return History(
        num_objectives=1,
        config_space=configspace_tiny
    )


@pytest.fixture
def history_single_obs(configspace_tiny):
    config_space = configspace_tiny
    # Create a History instance
    config1 = config_space.sample_configuration()
    config2 = config_space.sample_configuration()
    config3 = config_space.sample_configuration()
    config4 = config_space.sample_configuration()
    config5 = config_space.sample_configuration()
    config6 = config_space.sample_configuration()
    config7 = config_space.sample_configuration()
    config8 = config_space.sample_configuration()
    config9 = config_space.sample_configuration()
    config10 = config_space.sample_configuration()

    # Set up observations with different objective values
    observation1 = Observation(config=config1, objectives=[0.1], trial_state=SUCCESS)
    observation2 = Observation(config=config2, objectives=[0.3], trial_state=SUCCESS)
    observation3 = Observation(config=config3, objectives=[0.5], trial_state=SUCCESS)
    observation4 = Observation(config=config4, objectives=[2.0], trial_state=FAILED)
    observation5 = Observation(config=config5, objectives=[3.0], trial_state=SUCCESS)
    observation6 = Observation(config=config6, objectives=[10.5], trial_state=SUCCESS)
    observation7 = Observation(config=config7, objectives=[30.1], trial_state=SUCCESS)
    observation8 = Observation(config=config8, objectives=[100.5], trial_state=SUCCESS)
    observation9 = Observation(config=config9, objectives=[33.0], trial_state=SUCCESS)
    observation10 = Observation(config=config10, objectives=[12.5], trial_state=SUCCESS)

    history = History(
        num_objectives=1,
        config_space=configspace_tiny
    )

    # Add observations to history
    history.update_observations([observation1, observation2, observation3, observation4, observation5,
                                 observation6, observation7, observation8, observation9, observation10])

    return history


@pytest.fixture
def history_double(configspace_tiny):
    return History(
        num_objectives=2,
        config_space=configspace_tiny
    )


@pytest.fixture
def history_double_obs(configspace_tiny):
    config_space = configspace_tiny
    # Create a History instance
    config1 = config_space.sample_configuration()
    config2 = config_space.sample_configuration()
    config3 = config_space.sample_configuration()
    config4 = config_space.sample_configuration()

    # Set up observations with different objective values
    observation1 = Observation(config=config1, objectives=[0.1, 1.0], trial_state=SUCCESS)
    observation2 = Observation(config=config2, objectives=[0.5, 0.1], trial_state=SUCCESS)
    observation3 = Observation(config=config3, objectives=[0.5, 0.2], trial_state=SUCCESS)
    observation4 = Observation(config=config4, objectives=[2.0, 1.0], trial_state=FAILED)

    history = History(
        num_objectives=2,
        config_space=configspace_tiny
    )

    # Add observations to history
    history.update_observations([observation1, observation2, observation3, observation4])

    return history


@pytest.fixture
def history_double_cons(configspace_tiny):
    return History(
        num_objectives=2,
        num_constraints=2,
        config_space=configspace_tiny
    )


@pytest.fixture
def history_double_cons_obs(configspace_tiny):
    config_space = configspace_tiny
    # Create a History instance
    config1 = config_space.sample_configuration()
    config2 = config_space.sample_configuration()
    config3 = config_space.sample_configuration()
    config4 = config_space.sample_configuration()

    # Set up observations with different objective values
    observation1 = Observation(config=config1, objectives=[0.1, 1.0], constraints=[-0.1, -1.0], trial_state=SUCCESS)
    observation2 = Observation(config=config2, objectives=[0.5, 0.1], constraints=[-0.1, -1.0], trial_state=SUCCESS)
    observation3 = Observation(config=config3, objectives=[0.6, 0.2], constraints=[-0.6, -0.2], trial_state=SUCCESS)
    observation4 = Observation(config=config4, objectives=[2.0, 1.0], constraints=[-2.0, -1.0], trial_state=SUCCESS)
    history = History(
        num_objectives=2,
        num_constraints=2,
        config_space=configspace_tiny
    )

    history.update_observations([observation1, observation2, observation3, observation4])

    return history


@pytest.fixture
def history_4(configspace_tiny):
    return History(
        num_objectives=4,
        config_space=configspace_tiny
    )


@pytest.fixture
def multi_start_history_single_obs(configspace_tiny):
    config_space = configspace_tiny
    # Create a History instance
    config1 = config_space.sample_configuration()
    config2 = config_space.sample_configuration()
    config3 = config_space.sample_configuration()
    config4 = config_space.sample_configuration()

    # Set up observations with different objective values
    observation1 = Observation(config=config1, objectives=[0.1], trial_state=SUCCESS)
    observation2 = Observation(config=config2, objectives=[0.3], trial_state=SUCCESS)
    observation3 = Observation(config=config3, objectives=[0.5], trial_state=SUCCESS)
    observation4 = Observation(config=config4, objectives=[2.0], trial_state=FAILED)

    history = MultiStartHistory(
        num_objectives=1,
        config_space=configspace_tiny
    )

    # Add observations to history
    history.update_observations([observation1, observation2, observation3, observation4])

    return history


@pytest.fixture
def transfer_learning_history_single(history_single_obs):
    return [history_single_obs, history_single_obs, history_single_obs]


@pytest.fixture
def transfer_learning_history_double_cons(history_double_cons_obs):
    return [history_double_cons_obs, history_double_cons_obs, history_double_cons_obs]


@pytest.fixture
def multi_field_history_single_obs(history_single_obs):
    return [history_single_obs, history_single_obs, history_single_obs]
