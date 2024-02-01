import pytest
from openbox.utils.history import History, Observation, MultiStartHistory
from datetime import datetime
from ConfigSpace import Configuration, ConfigurationSpace
from openbox.utils.constants import SUCCESS, FAILED
import matplotlib.pyplot as plt
import numpy as np
import copy


class TestObservation:
    @pytest.fixture
    def config(self, configspace_tiny):
        cs = configspace_tiny
        return Configuration(cs, {'x1': 0, 'x2': 0})

    @pytest.fixture
    def objectives(self):
        return [1.0, 2.0, 3.0]

    @pytest.fixture
    def constraints(self):
        return [0.1, 0.2, 0.3]

    @pytest.fixture
    def extra_info(self):
        return {'key': 'value'}

    @pytest.fixture
    def observation(self, config, objectives, constraints, extra_info):
        return Observation(
            config=config,
            objectives=objectives,
            constraints=constraints,
            extra_info=extra_info,
            trial_state=SUCCESS,
            elapsed_time=10.0
        )

    def test_observation_init(self, observation, config, objectives, constraints, extra_info):
        assert observation.config == config
        assert observation.objectives == objectives
        assert observation.constraints == constraints
        assert observation.extra_info == extra_info
        assert observation.trial_state == SUCCESS
        assert observation.elapsed_time == 10.0
        assert isinstance(observation.create_time, datetime)

    def test_observation_str_repr(self, observation):
        assert str(observation) == repr(observation)

    def test_observation_to_dict(self, observation):
        observation_dict = observation.to_dict()
        assert isinstance(observation_dict, dict)
        assert 'config' in observation_dict
        assert 'objectives' in observation_dict
        assert 'constraints' in observation_dict
        assert 'trial_state' in observation_dict
        assert 'elapsed_time' in observation_dict
        assert 'create_time' in observation_dict
        assert 'extra_info' in observation_dict

    def test_observation_from_dict(self, observation, configspace_tiny):
        cs = configspace_tiny
        observation_dict = observation.to_dict()
        new_observation = Observation.from_dict(observation_dict, cs)
        assert new_observation == observation

    def test_observation_equality(self, observation):
        new_observation = copy.deepcopy(observation)
        assert observation == new_observation

    def test_observation_equality_different_objectives(self, observation):
        new_observation = copy.deepcopy(observation)
        new_observation.objectives = [1.0, 2.0, 4.0]
        assert observation != new_observation

    def test_observation_equality_different_constraints(self, observation):
        new_observation = copy.deepcopy(observation)
        new_observation.constraints = [0.1, 0.2, 0.4]
        assert observation != new_observation

    def test_observation_equality_different_extra_info(self, observation):
        new_observation = copy.deepcopy(observation)
        new_observation.extra_info = {'key': 'new_value'}
        assert observation != new_observation


class TestHistory:

    def test_history_init(self, history_double):
        assert history_double.task_id == 'OpenBox'
        assert history_double.num_objectives == 2
        assert history_double.num_constraints == 0
        assert history_double.config_space is not None
        assert history_double.meta_info == {}
        assert history_double.observations == []
        assert isinstance(history_double.global_start_time, datetime)
        assert history_double._ref_point is None

    def test_history_len(self, history_double, configspace_tiny):
        assert len(history_double) == 0
        history_double.observations.append(Observation(configspace_tiny, [1.0]))
        assert len(history_double) == 1

    def test_history_empty(self, history_double, configspace_tiny):
        assert history_double.empty()
        history_double.observations.append(Observation(configspace_tiny, [1.0]))
        assert not history_double.empty()

    def test_history_configurations(self, history_double, configspace_tiny):
        config = configspace_tiny
        history_double.observations.append(Observation(config, [1.0]))
        assert history_double.configurations == [config]

    def test_history_objectives(self, history_double, configspace_tiny):
        objectives = [1.0]
        history_double.observations.append(Observation(configspace_tiny.sample_configuration(), objectives))
        assert history_double.objectives == [objectives]

    def test_history_constraints(self, history_double_cons, configspace_tiny):
        constraints = [0.1, 0.2]
        history_double_cons.observations.append(Observation(configspace_tiny.sample_configuration(), [1.0, 2.0], constraints))
        assert history_double_cons.constraints == [constraints]

    def test_history_trial_states(self, history_double, configspace_tiny):
        history_double.observations.append(
            Observation(configspace_tiny.sample_configuration(), [1.0, 3.0], trial_state=SUCCESS))
        assert history_double.trial_states == [SUCCESS]

    def test_history_elapsed_times(self, history_double, configspace_tiny):
        history_double.observations.append(Observation(configspace_tiny.sample_configuration(), [1.0, 1.0], elapsed_time=10.0))
        assert history_double.elapsed_times == [10.0]

    def test_history_create_times(self, history_double, configspace_tiny):
        now = datetime.now()
        history_double.observations.append(Observation(configspace_tiny, [1.0, 2.0]))
        assert len(history_double.create_times) == 1

    def test_history_extra_infos(self, history_double, configspace_tiny):
        extra_info = {'key': 'value'}
        history_double.observations.append(Observation(configspace_tiny, [1.0, 3.0], extra_info=extra_info))
        assert history_double.extra_infos == [extra_info]

    def test_history_ref_point(self, history_double):
        ref_point = [0.1, 0.2]
        history_double.ref_point = ref_point
        assert history_double.ref_point == ref_point

    def test_check_ref_point_valid_list(self, history_double):
        ref_point = [0.1, 0.2]
        result = history_double.check_ref_point(ref_point)
        assert result == ref_point

    def test_is_valid_observation_valid_observation(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        config = config_space.sample_configuration()
        observation = Observation(config, [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        result = history_double.is_valid_observation(observation)
        assert result is True

        with pytest.raises(ValueError, match="observation must be an instance of Observation, got <class 'str'>"):
            history_double.is_valid_observation("invalid_observation")

        observation = Observation("invalid_config", [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        with pytest.raises(ValueError, match="config must be an instance of Configuration, got <class 'str'>"):
            history_double.is_valid_observation(observation)

        observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        with pytest.raises(ValueError, match="num objectives must be 2, got 1"):
            history_double.is_valid_observation(observation)

        observation = Observation(config, [0.1, np.nan], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        with pytest.raises(ValueError,
                           match="invalid values .* are not allowed in objectives in a SUCCESS trial, got .*"):
            history_double.is_valid_observation(observation)

    def test_update_observation_valid_observation(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        config = config_space.sample_configuration()
        observation = Observation(config, [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        history_double.update_observation(observation)
        assert len(history_double.observations) == 1
        assert history_double.observations[0] == observation

    def test_update_observations_valid_observations(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()
        observation1 = Observation(config1, [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        observation2 = Observation(config2, [0.2, 0.3], trial_state=SUCCESS, elapsed_time=3.0, extra_info={})
        observations = [observation1, observation2]
        history_double.update_observations(observations)
        assert len(history_double.observations) == 2
        assert history_double.observations == observations

    def test_get_config_space_with_config_space(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        assert history_double.get_config_space() == config_space

        history_double.config_space = None
        assert history_double.get_config_space() is None

        config = config_space.sample_configuration()
        observation = Observation(config, [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        history_double.update_observation(observation)

        assert history_double.get_config_space() == config_space

    def test_get_config_array(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()
        observation1 = Observation(config1, [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        observation2 = Observation(config2, [0.2, 0.3], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        history_double.update_observations([observation1, observation2])
        config_array = history_double.get_config_array(transform='scale')
        assert config_array.shape == (2, 2)
        assert np.all(config_array >= 0) and np.all(config_array <= 1)

        config_array = history_double.get_config_array(transform='numerical')
        assert config_array.shape == (2, 2)
        assert np.all(config_array == config_array.astype(float))

    def test_get_config_dicts(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        config = config_space.sample_configuration()
        observation = Observation(config, [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        history_double.update_observation(observation)
        config_dicts = history_double.get_config_dicts()
        assert len(config_dicts) == 1
        assert config_dicts[0] == config.get_dictionary()

    def test_get_min_max_values(self, history_double):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [np.nan, np.inf, -np.inf]])
        min_values, max_values = history_double._get_min_max_values(x, axis=0)
        assert np.all(min_values == [1.0, 2.0, 3.0]) and np.all(max_values == [4.0, 5.0, 6.0])

    def test_get_transformed_values_none_transform(self, history_double, history_double_cons, configspace_tiny):
        config_space = configspace_tiny
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()
        observation1 = Observation(config1, [0.1, 0.2], trial_state=FAILED, elapsed_time=2.0, extra_info={})
        observation2 = Observation(config2, [0.2, 0.3], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        history_double.update_observations([observation1, observation2])

        transformed_values = history_double._get_transformed_values('objectives', transform='none')
        assert np.all(transformed_values == np.array([[0.1, 0.2], [0.2, 0.3]]))

        transformed_values = history_double._get_transformed_values('objectives', transform='failed')
        assert np.all(transformed_values == np.array([[0.2, 0.3], [0.2, 0.3]]))

        transformed_values = history_double._get_transformed_values('objectives', transform='failed,none')
        assert np.all(np.isfinite(transformed_values))  # check if all values are finite

        observation3 = Observation(config1, [0.1, 0.2], [-0.1, -0.2], trial_state=FAILED, elapsed_time=2.0,
                                   extra_info={})
        observation4 = Observation(config2, [0.2, 0.3], [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0,
                                   extra_info={})
        history_double_cons.update_observations([observation3, observation4])

        transformed_values = history_double_cons._get_transformed_values('objectives', transform='infeasible')
        assert np.all(transformed_values == np.array([[0.2, 0.3], [0.2, 0.3]]))

        with pytest.raises(ValueError, match='Cannot use "infeasible" transform for constraints!'):
            history_double_cons._get_transformed_values('constraints', transform='infeasible')

    def test_get_objectives(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()
        observation1 = Observation(config1, [0.1, 0.2], trial_state=FAILED, elapsed_time=2.0, extra_info={})
        observation2 = Observation(config2, [0.2, 0.3], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        history_double.update_observations([observation1, observation2])
        objectives = history_double.get_objectives(transform='failed,none')
        assert np.all(objectives == np.array([[0.2, 0.3], [0.2, 0.3]]))

    def test_get_constraints(self, history_double_cons, configspace_tiny):
        config_space = configspace_tiny
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()
        observation3 = Observation(config1, [0.1, 0.2], [0.1, 0.2], trial_state=FAILED, elapsed_time=2.0, extra_info={})
        observation4 = Observation(config2, [0.2, 0.3], [-0.1, -0.2], trial_state=SUCCESS, elapsed_time=2.0,
                                   extra_info={})
        history_double_cons.update_observations([observation3, observation4])

        constraints = history_double_cons.get_constraints(transform='none')
        assert np.all(constraints == np.array([[0.1, 0.2], [-0.1, -0.2]]))

    def test_get_success_mask_count(self, history_double, configspace_tiny):
        config_space = configspace_tiny
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()
        observation1 = Observation(config1, [0.1, 0.2], trial_state=FAILED, elapsed_time=2.0, extra_info={})
        observation2 = Observation(config2, [0.2, 0.3], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        history_double.update_observations([observation1, observation2])

        success_mask = history_double.get_success_mask()
        assert np.all(success_mask == np.array([False, True]))
        assert len(success_mask) == 2

    def test_get_feasible_mask_count(self, history_double_cons, configspace_tiny):
        config_space = configspace_tiny
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()
        observation1 = Observation(config1, [0.1, 0.2], [0.1, 0.2], trial_state=SUCCESS, elapsed_time=2.0,
                                   extra_info={})
        observation2 = Observation(config2, [0.2, 0.3], [-0.1, -0.2], trial_state=SUCCESS, elapsed_time=2.0,
                                   extra_info={})
        history_double_cons.update_observations([observation1, observation2])

        feasible_mask = history_double_cons.get_feasible_mask(exclude_failed=True)
        assert np.all(feasible_mask == np.array([False, True]))
        assert len(feasible_mask) == 2

    def test_get_incumbents(self, history_double, history_single_obs):
        # Calculate incumbents
        incumbents = history_single_obs.get_incumbents()

        # Check if the returned incumbents match the expected observations
        expected_incumbents = [history_single_obs.observations[0]]
        assert incumbents == expected_incumbents

        with pytest.raises(ValueError):
            history_double.get_incumbents()

    def test_get_incumbent_value(self, history_double, history_single_obs):
        # Calculate incumbents
        incumbents = history_single_obs.get_incumbent_value()

        # Check if the returned incumbents match the expected value
        assert incumbents == 0.1

        with pytest.raises(ValueError):
            history_double.get_incumbent_value()

    def test_get_incumbent_configs(self, history_double, history_single_obs):
        # Calculate incumbents
        incumbents = history_single_obs.get_incumbent_configs()

        # Check if the returned incumbents match the expected configs
        assert incumbents == [history_single_obs.observations[0].config]

        with pytest.raises(ValueError):
            history_double.get_incumbent_configs()

    def test_get_mo_incumbent_values(self, history_double_obs, history_single):
        # Calculate incumbents
        incumbents = history_double_obs.get_mo_incumbent_values()

        # Check if the returned incumbents match the expected configs
        assert np.all(incumbents == [0.1, 0.1])

        with pytest.raises(AssertionError):
            history_single.get_mo_incumbent_values()

    def test_get_pareto(self, history_double_obs, history_single):
        # Calculate incumbents
        paretos = history_double_obs.get_pareto()

        # Check if the returned incumbents match the expected configs
        expected_paretos = history_double_obs.observations[:2]
        assert paretos == expected_paretos

        with pytest.raises(AssertionError):
            history_single.get_pareto()

    def test_get_pareto_front(self, history_double_obs, history_single):
        # Calculate incumbents
        paretos = history_double_obs.get_pareto_front()

        # Check if the returned incumbents match the expected configs
        assert np.all(paretos == np.array([[0.5, 0.1], [0.1, 1.0]]))

        with pytest.raises(AssertionError):
            history_single.get_pareto_front()

    def test_get_pareto_set(self, history_double_obs, history_single):
        # Calculate incumbents
        paretos = history_double_obs.get_pareto_set()

        # Check if the returned incumbents match the expected configs
        expected_paretos = [ob.config for ob in history_double_obs.observations[:2]]
        assert paretos == expected_paretos

        with pytest.raises(AssertionError):
            history_single.get_pareto_set()

    def test_compute_hypervolume(self, history_double_obs, history_single):
        with pytest.raises(AssertionError):
            history_double_obs.compute_hypervolume()

        hv = history_double_obs.compute_hypervolume([3.0, 3.0], data_range='last')
        assert hv == 8.05

        hvs = history_double_obs.compute_hypervolume([3.0, 3.0], data_range='all')
        assert np.all(hvs == [5.8, 8.05, 8.05, 8.05])

        with pytest.raises(ValueError):
            history_double_obs.compute_hypervolume([3.0, 3.0], data_range='invalid data range')

        with pytest.raises(AssertionError):
            history_single.compute_hypervolume()

    def test_get_str(self, history_double_cons_obs):
        assert history_double_cons_obs.get_str() is not None

        assert str(history_double_cons_obs) == history_double_cons_obs.get_str()

    def test_get_importance(self, history_double_cons_obs):
        assert isinstance(history_double_cons_obs.get_importance(method='fanova', return_dict=True), dict)
        assert isinstance(history_double_cons_obs.get_importance(method='shap', return_dict=True), dict)
        from prettytable import PrettyTable
        assert isinstance(history_double_cons_obs.get_importance(method='fanova', return_dict=False), PrettyTable)

    def test_plot_convergence(self, history_double, history_single_obs):
        assert isinstance(history_single_obs.plot_convergence(), plt.Axes)

        with pytest.raises(ValueError):
            history_double.plot_convergence()

    def test_plot_pareto_front(self, history_double_obs, history_single, history_4):
        assert isinstance(history_double_obs.plot_pareto_front(), plt.Axes)

        with pytest.raises(AssertionError):
            history_single.plot_pareto_front()

        with pytest.raises(ValueError, match='plot_pareto_front only supports 2 or 3 objectives!'):
            history_4.plot_pareto_front()

    def test_plot_hypervolumes(self, history_double_obs, history_single_obs):
        with pytest.raises(AssertionError):
            history_double_obs.plot_hypervolumes()

        assert isinstance(history_double_obs.plot_hypervolumes(ref_point=[3.0, 3.0]), plt.Axes)

        with pytest.raises(AssertionError):
            history_single_obs.plot_hypervolumes()

    def test_visualize_html(self, history_double_obs):
        from openbox.visualization import HTMLVisualizer

        assert isinstance(history_double_obs.visualize_html(), HTMLVisualizer)

    def test_visualize_hiplot(self, history_double_obs):
        with pytest.raises(RuntimeError,
                           match="`display` can only be called on an ipython context. Are you in a notebook?"):
            history_double_obs.visualize_hiplot()

    def test_save_json(self, history_double_obs):
        history_double_obs.save_json('test/datas/tmp.json')

    def test_load_json(self, history_double_obs, configspace_tiny):
        with pytest.raises(FileNotFoundError):
            history_double_obs.load_json('test/datas/notfound.json', config_space=configspace_tiny)

        new_history = history_double_obs.load_json('test/datas/tmp.json', config_space=configspace_tiny)

        assert new_history is not None


def test_get_observations_for_all_restarts(configspace_tiny):
    config_space = configspace_tiny  # Replace with your actual ConfigurationSpace instance
    example_multistart_history = MultiStartHistory(
        task_id='TestTask',
        num_objectives=1,
        num_constraints=0,
        config_space=config_space,
        meta_info={'key': 'value'}
    )
    # Test the get_observations_for_all_restarts method
    config1 = example_multistart_history.config_space.sample_configuration()
    obs1 = Observation(config1, objectives=[0.5], trial_state='SUCCESS', constraints=None, extra_info={'key': 'value'})
    config2 = example_multistart_history.config_space.sample_configuration()
    obs2 = Observation(config2, objectives=[0.8], trial_state='SUCCESS', constraints=None, extra_info={'key': 'value'})

    example_multistart_history.update_observations([obs1, obs2])
    example_multistart_history.restart()
    assert len(example_multistart_history.stored_observations) == 1
    assert len(example_multistart_history.observations) == 0

    config3 = example_multistart_history.config_space.sample_configuration()
    obs3 = Observation(config3, objectives=[0.7], trial_state='SUCCESS', constraints=None, extra_info={'key': 'value'})
    config4 = example_multistart_history.config_space.sample_configuration()
    obs4 = Observation(config4, objectives=[0.6], trial_state='SUCCESS', constraints=None, extra_info={'key': 'value'})

    example_multistart_history.update_observations([obs3, obs4])

    all_observations = example_multistart_history.get_observations_for_all_restarts()
    assert len(all_observations) == 4
    assert obs1 in all_observations
    assert obs2 in all_observations
    assert obs3 in all_observations
    assert obs4 in all_observations
