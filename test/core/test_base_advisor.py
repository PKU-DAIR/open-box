import unittest
from unittest.mock import Mock
from openbox.core.base_advisor import BaseAdvisor
from ConfigSpace import ConfigurationSpace, Configuration
from openbox.utils.history import History, Observation
from openbox.utils.constants import MAXINT, SUCCESS


class BaseAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.config_space = ConfigurationSpace()
        self.history = History(task_id='test', num_objectives=1, num_constraints=0, config_space=self.config_space)
        self.advisor = BaseAdvisor(config_space=self.config_space, task_id='test')

    def test_advisor_get_suggestion_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.advisor.get_suggestion()

    def test_advisor_get_suggestions_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.advisor.get_suggestions()

    def test_advisor_update_observation_updates_history(self):
        observation = Observation(config=self.config_space.sample_configuration(), trial_state=SUCCESS, objs=[1.0])
        self.advisor.update_observation(observation)
        self.assertEqual(self.advisor.history.observations[-1], observation)

    def test_advisor_update_observations_updates_history(self):
        observations = [Observation(config=self.config_space.sample_configuration(), trial_state=SUCCESS, objs=[1.0]) for _ in range(5)]
        self.advisor.update_observations(observations)
        self.assertEqual(self.advisor.history.observations[-5:], observations)

    def test_advisor_sample_random_configs_returns_configs(self):
        configs = self.advisor.sample_random_configs(self.config_space, num_configs=5)
        self.assertEqual(len(configs), 5)
        for config in configs:
            self.assertIsInstance(config, Configuration)

    def test_advisor_get_history_returns_history(self):
        history = self.advisor.get_history()
        self.assertEqual(history.task_id, self.history.task_id)

if __name__ == '__main__':
    unittest.main()