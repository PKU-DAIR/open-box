# import pytest
# from unittest.mock import MagicMock, patch
# from openbox.optimizer.base import BOBase
# from openbox.optimizer.message_queue_smbo import mqSMBO
# from openbox.utils.config_space import ConfigurationSpace
# from openbox.utils.history import History, Observation
#
#
# def test_mqsmbo_initialization(configspace_tiny, func_brain):
#     config_space = configspace_tiny
#     objective_function = func_brain
#     mqsmbo = mqSMBO(objective_function, config_space, initial_runs=1, max_runs=2)
#     assert isinstance(mqsmbo, BOBase)
#     assert mqsmbo.objective_function == objective_function
#     assert mqsmbo.config_space == config_space
#
#
# def test_mqsmbo_init_with_different_parallel_strategy(configspace_tiny, func_brain):
#     config_space = configspace_tiny
#     objective_function = func_brain
#     mqsmbo = mqSMBO(objective_function, config_space, parallel_strategy='sync')
#     assert mqsmbo.config_advisor is not None
#     mqsmbo = mqSMBO(objective_function, config_space, parallel_strategy='async')
#     assert mqsmbo.config_advisor is not None
#     with pytest.raises(ValueError):
#         mqSMBO(objective_function, config_space, parallel_strategy='invalid')
#
#
# def test_mqsmbo_async_run(configspace_tiny, func_brain):
#     config_space = configspace_tiny
#     objective_function = func_brain
#     mqsmbo = mqSMBO(objective_function, config_space, initial_runs=1, max_runs=2)
#     mqsmbo.config_advisor.get_suggestion = MagicMock()
#     mqsmbo.master_messager.receive_message = MagicMock()
#     mqsmbo.async_run()
#     assert mqsmbo.config_advisor.get_suggestion.call_count == 2
#     assert mqsmbo.master_messager.receive_message.call_count == 2
#
#
# def test_mqsmbo_sync_run(configspace_tiny, func_brain):
#     config_space = configspace_tiny
#     objective_function = func_brain
#     mqsmbo = mqSMBO(objective_function, config_space, initial_runs=1, max_runs=2)
#     mqsmbo.config_advisor.get_suggestions = MagicMock(return_value=[1, 2])
#     mqsmbo.master_messager.receive_message = MagicMock()
#     mqsmbo.sync_run()
#     assert mqsmbo.config_advisor.get_suggestions.call_count == 2
#     assert mqsmbo.master_messager.receive_message.call_count == 2
#
#
# def test_mqsmbo_run(configspace_tiny, func_brain):
#     config_space = configspace_tiny
#     objective_function = func_brain
#     mqsmbo = mqSMBO(objective_function, config_space, initial_runs=1, max_runs=2)
#     with patch.object(mqSMBO, 'async_run') as mock_async_run, patch.object(mqSMBO, 'sync_run') as mock_sync_run:
#         mqsmbo.run()
#         mock_async_run.assert_called_once()
#         mock_sync_run.assert_not_called()
#         mqsmbo.parallel_strategy = 'sync'
#         mqsmbo.run()
#         mock_sync_run.assert_called_once()
