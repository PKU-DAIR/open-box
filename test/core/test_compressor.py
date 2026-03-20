import pytest

from openbox.core.generic_advisor import Advisor
from openbox.core.space_adapter import IdentitySpaceAdapter, CompressorSpaceAdapter
from openbox.compressor import Compressor
from openbox.compressor.api.step_factory import create_steps_from_strings
from openbox.utils.constants import SUCCESS
from openbox.utils.history import Observation


def test_generic_advisor_without_compressor_uses_identity_adapter(configspace_tiny):
    advisor = Advisor(configspace_tiny, initial_trials=1)
    assert isinstance(advisor.space_adapter, IdentitySpaceAdapter)
    assert advisor.sample_space == configspace_tiny
    assert advisor.surrogate_space == configspace_tiny


def test_generic_advisor_with_compressor_adapter(configspace_tiny):
    advisor = Advisor(
        config_space=configspace_tiny,
        initial_trials=1,
        compressor_type='none', # no compressor, use identity space adapter
        compressor_kwargs={},
    )
    assert isinstance(advisor.space_adapter, IdentitySpaceAdapter)
    assert advisor.sample_space == configspace_tiny
    assert advisor.surrogate_space == configspace_tiny

    config = advisor.get_suggestion()
    observation = Observation(
        config=config,
        objectives=[0.1],
        trial_state=SUCCESS,
        elapsed_time=0.1,
        extra_info={},
    )
    advisor.update_observation(observation)
    assert len(advisor.history) == 1


def test_compressor_adapter_roundtrip_and_spaces(configspace_tiny):
    advisor = Advisor(
        config_space=configspace_tiny,
        initial_trials=1,
        compressor_type='pipeline',
        compressor_kwargs={
            'step_strings': ['p_quant'],
            'step_params': {'p_quant': {'max_num_values': 4, 'seed': 3}},
            'seed': 3,
        },
    )
    adapter = advisor.space_adapter
    config = advisor.get_suggestion()

    surrogate_config = adapter.config_to_surrogate(config)
    sample_config = adapter.config_to_sample(config)
    original_config = adapter.config_to_original(sample_config)

    surrogate_dict = surrogate_config.get_dictionary()
    sample_dict = sample_config.get_dictionary()
    original_dict = original_config.get_dictionary()

    assert set(surrogate_dict.keys()) == {'x1|q', 'x2|q'}
    assert set(sample_dict.keys()) == {'x1|q', 'x2|q'}
    assert all(1 <= int(v) <= 4 for v in surrogate_dict.values())
    assert all(1 <= int(v) <= 4 for v in sample_dict.values())

    # unproject returns original-space parameter names
    assert set(original_dict.keys()) == {'x1', 'x2'}
    assert advisor.sample_space != advisor.config_space
    assert advisor.surrogate_space != advisor.config_space


def test_compressor_adapter_history_and_update_compression(configspace_tiny):
    advisor = Advisor(
        config_space=configspace_tiny,
        initial_trials=1,
        compressor_type='none',
        compressor_kwargs={},
    )
    adapter = advisor.space_adapter
    assert isinstance(adapter, IdentitySpaceAdapter)
    config = advisor.get_suggestion()
    observation = Observation(
        config=config,
        objectives=[0.2],
        trial_state=SUCCESS,
        elapsed_time=0.1,
        extra_info={},
    )
    advisor.update_observation(observation)
    acq_history = adapter.history_for_acq(advisor.history)  # identity adapter keeps acquisition history untouched
    assert acq_history is advisor.history
    # no adaptive update for identity adapter
    assert adapter.update(advisor.history) is False
    assert advisor.update_compression(advisor.history) is False


def test_compressor_adapter_caches_low_dim_config(configspace_tiny):
    advisor = Advisor(
        config_space=configspace_tiny,
        initial_trials=1,
        compressor_type='pipeline',
        compressor_kwargs={
            'step_strings': ['p_quant'],
            'step_params': {'p_quant': {'max_num_values': 4, 'seed': 5}},
            'seed': 5,
        },
    )
    config = advisor.get_suggestion()
    config._low_dim_config = {'x1': 0.0, 'x2': 1.0}

    observation = Observation(
        config=config,
        objectives=[0.3],
        trial_state=SUCCESS,
        elapsed_time=0.1,
        extra_info=None,
    )
    advisor.update_observation(observation)

    assert len(advisor.history) == 1
    assert 'low_dim_config' in advisor.history.observations[0].extra_info
    assert advisor.history.observations[0].extra_info['low_dim_config'] == {'x1': 0.0, 'x2': 1.0}


def test_compressor_adapter_with_transfer_learning_history(configspace_tiny, transfer_learning_history_single):
    advisor = Advisor(
        config_space=configspace_tiny,
        initial_trials=1,
        compressor_type='pipeline',
        compressor_kwargs={
            'step_strings': ['p_quant'],
            'step_params': {'p_quant': {'max_num_values': 4, 'seed': 6}},
            'seed': 6,
        },
        transfer_learning_history=transfer_learning_history_single,
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.surrogate_transfer_learning_history is not None
    assert len(advisor.surrogate_transfer_learning_history) == len(transfer_learning_history_single)
    assert all(
        h.config_space == advisor.surrogate_space
        for h in advisor.surrogate_transfer_learning_history
    )


def test_generic_advisor_with_llamatune_quantization(configspace_huge):
    advisor = Advisor(
        config_space=configspace_huge,
        initial_trials=1,
        compressor_type='llamatune',
        compressor_kwargs={
            'adapter_alias': 'none',
            'max_num_values': 5,
            'seed': 1,
        },
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert len(advisor.surrogate_space.get_hyperparameters()) == len(configspace_huge.get_hyperparameters())
    assert 'x1|q' in advisor.surrogate_space.get_hyperparameter_names()
    assert 'x2|q' in advisor.surrogate_space.get_hyperparameter_names()
    assert 'x4|q' in advisor.surrogate_space.get_hyperparameter_names()
    assert 'x1' not in advisor.surrogate_space.get_hyperparameter_names()
    assert 'x2' not in advisor.surrogate_space.get_hyperparameter_names()
    assert 'x4' not in advisor.surrogate_space.get_hyperparameter_names()

    config = advisor.get_suggestion()
    observation = Observation(
        config=config,
        objectives=[0.15],
        trial_state=SUCCESS,
        elapsed_time=0.1,
        extra_info={},
    )
    advisor.update_observation(observation)
    assert len(advisor.history) == 1


def test_generic_advisor_with_shap_dimension_compression(configspace_tiny, transfer_learning_history_single):
    pytest.importorskip('shap')

    advisor = Advisor(
        config_space=configspace_tiny,
        initial_trials=1,
        compressor_type='shap',
        compressor_kwargs={
            'strategy': 'shap',
            'topk': 1,
            'top_ratio': 1.0,
            'sigma': 0.0,
            'seed': 2,
        },
        transfer_learning_history=transfer_learning_history_single,
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.space_adapter.compressor.pipeline is not None
    assert any(type(step).__name__ == 'SHAPDimensionStep'
               for step in advisor.space_adapter.compressor.pipeline.steps)
    assert len(advisor.surrogate_space.get_hyperparameters()) == 1
    assert advisor.surrogate_space.get_hyperparameter_names()[0] in ['x1', 'x2']

    config = advisor.get_suggestion()
    observation = Observation(
        config=config,
        objectives=[0.11],
        trial_state=SUCCESS,
        elapsed_time=0.1,
        extra_info={},
    )
    advisor.update_observation(observation)
    assert len(advisor.history) == 1


def test_generic_advisor_with_custom_pipeline_steps(configspace_huge):
    steps = create_steps_from_strings(
        ['p_quant'],
        step_params={'p_quant': {'max_num_values': 4, 'seed': 3}},
    )
    advisor = Advisor(
        config_space=configspace_huge,
        initial_trials=1,
        compressor_type='pipeline',
        compressor_kwargs={
            'steps': steps,
            'seed': 3,
        },
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.space_adapter.compressor.pipeline is not None
    assert len(advisor.space_adapter.compressor.pipeline.steps) == 1
    assert 'x4|q' in advisor.surrogate_space.get_hyperparameter_names()


def test_generic_advisor_with_step_strings_api_build(configspace_huge):
    advisor = Advisor(
        config_space=configspace_huge,
        initial_trials=1,
        compressor_type='pipeline',
        compressor_kwargs={
            'step_strings': ['p_quant'],
            'step_params': {'p_quant': {'max_num_values': 3, 'seed': 9}},
            'seed': 9,
        },
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.space_adapter.compressor.pipeline is not None
    assert len(advisor.space_adapter.compressor.pipeline.steps) == 1
    assert advisor.space_adapter.compressor.pipeline.steps[0].__class__.__name__ == 'QuantizationProjectionStep'
    assert 'x4|q' in advisor.surrogate_space.get_hyperparameter_names()


def test_generic_advisor_with_prebuilt_compressor_instance(configspace_huge):
    steps = create_steps_from_strings(
        ['p_quant'],
        step_params={'p_quant': {'max_num_values': 6, 'seed': 7}},
    )
    compressor = Compressor(
        config_space=configspace_huge,
        steps=steps,
        seed=7,
    )
    advisor = Advisor(
        config_space=configspace_huge,
        initial_trials=1,
        compressor=compressor,
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.space_adapter.compressor is compressor
    assert 'x4|q' in advisor.surrogate_space.get_hyperparameter_names()


def test_generic_advisor_with_expert_dimension_compression(configspace_huge):
    advisor = Advisor(
        config_space=configspace_huge,
        initial_trials=1,
        compressor_type='expert',
        compressor_kwargs={
            'expert_params': ['x1', 'x4'],
            'top_ratio': 1.0,
            'sigma': 0.0,
            'seed': 11,
        },
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.space_adapter.compressor.pipeline is not None
    assert any(type(step).__name__ == 'ExpertDimensionStep'
               for step in advisor.space_adapter.compressor.pipeline.steps)
    assert set(advisor.surrogate_space.get_hyperparameter_names()) == {'x1', 'x4'}


def test_generic_advisor_with_llamatune_rembo_projection(configspace_huge):
    advisor = Advisor(
        config_space=configspace_huge,
        initial_trials=1,
        compressor_type='llamatune',
        compressor_kwargs={
            'adapter_alias': 'rembo',
            'le_low_dim': 2,
            'max_num_values': 6,
            'seed': 13,
        },
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.space_adapter.compressor.pipeline is not None
    assert advisor.space_adapter.compressor.needs_unproject()
    assert len(advisor.sample_space.get_hyperparameters()) == 2
    assert all(name.startswith('rembo_') for name in advisor.sample_space.get_hyperparameter_names())


def test_generic_advisor_with_pipeline_steps_as_strings(configspace_huge):
    advisor = Advisor(
        config_space=configspace_huge,
        initial_trials=1,
        compressor_type='pipeline',
        compressor_kwargs={
            'steps': ['p_quant'],
            'step_params': {'p_quant': {'max_num_values': 5, 'seed': 17}},
            'seed': 17,
        },
    )

    assert isinstance(advisor.space_adapter, CompressorSpaceAdapter)
    assert advisor.space_adapter.compressor.pipeline is not None
    assert advisor.space_adapter.compressor.pipeline.steps[0].__class__.__name__ == 'QuantizationProjectionStep'
    assert 'x4|q' in advisor.surrogate_space.get_hyperparameter_names()


@pytest.mark.parametrize(
    'step_strings,step_params,expected_unprojected_names,uses_projection',
    [
        (
            ['d_expert', 'r_boundary'],
            {
                'd_expert': {'expert_params': ['x1']},
                'r_boundary': {'top_ratio': 0.8, 'sigma': 0.0},
            },
            {'x1'},
            False,
        ),
        (
            ['d_expert', 'r_boundary', 'p_rembo'],
            {
                'd_expert': {'expert_params': ['x1']},
                'r_boundary': {'top_ratio': 0.8, 'sigma': 0.0},
                'p_rembo': {'low_dim': 1, 'seed': 19},
            },
            {'x1'},
            True,
        ),
        (
            ['d_expert', 'p_rembo'],
            {
                'd_expert': {'expert_params': ['x1']},
                'p_rembo': {'low_dim': 1, 'seed': 23},
            },
            {'x1'},
            True,
        ),
        (
            ['r_boundary', 'p_rembo'],
            {
                'r_boundary': {'top_ratio': 0.8, 'sigma': 0.0},
                'p_rembo': {'low_dim': 1, 'seed': 29},
            },
            {'x1', 'x2'},
            True,
        ),
    ],
    ids=[
        'dimension+boundary',
        'dimension+boundary+projection',
        'dimension+projection',
        'boundary+projection',
    ],
)
def test_generic_advisor_pipeline_step_combinations(
    configspace_tiny,
    transfer_learning_history_single,
    step_strings,
    step_params,
    expected_unprojected_names,
    uses_projection,
):
    advisor = Advisor(
        config_space=configspace_tiny,
        initial_trials=1,
        compressor_type='pipeline',
        compressor_kwargs={
            'step_strings': step_strings,
            'step_params': step_params,
            'seed': 31,
        },
        transfer_learning_history=transfer_learning_history_single,
    )

    pipeline = advisor.space_adapter.compressor.pipeline
    assert pipeline is not None
    assert [step.__class__.__name__ for step in pipeline.steps] == [
        'ExpertDimensionStep' if s == 'd_expert'
        else 'BoundaryRangeStep' if s == 'r_boundary'
        else 'REMBOProjectionStep'
        for s in step_strings
    ]

    config = advisor.get_suggestion()
    surrogate_config = advisor.space_adapter.config_to_surrogate(config)
    surrogate_names = set(surrogate_config.get_dictionary().keys())

    if uses_projection:
        assert all(name.startswith('rembo_') for name in surrogate_names)
    else:
        assert surrogate_names == expected_unprojected_names

    unprojected_names = set(advisor.space_adapter.compressor.unprojected_space.get_hyperparameter_names())
    assert unprojected_names == expected_unprojected_names

    observation = Observation(
        config=config,
        objectives=[0.17],
        trial_state=SUCCESS,
        elapsed_time=0.1,
        extra_info={},
    )
    advisor.update_observation(observation)
    assert len(advisor.history) == 1


def test_pipeline_type_requires_steps_or_step_strings(configspace_huge):
    with pytest.raises(ValueError, match='requires `steps` or `step_strings`'):
        Advisor(
            config_space=configspace_huge,
            initial_trials=1,
            compressor_type='pipeline',
            compressor_kwargs={},
        )
