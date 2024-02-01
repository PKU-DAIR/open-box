import pytest
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.samplers import Sampler, SobolSampler, LatinHypercubeSampler, HaltonSampler
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter


def test_sampler_initialization_with_valid_input():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(UniformFloatHyperparameter("param1", 0.0, 1.0))
    sampler = Sampler(config_space, 10)
    assert sampler is not None


def test_sampler_initialization_with_invalid_input():
    with pytest.raises(NotImplementedError):
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(UniformIntegerHyperparameter("param1", 0, 10))
        sampler = Sampler(config_space, 10)
        sampler.generate()


def test_sobol_sampler_generation():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(UniformFloatHyperparameter("param1", 0.0, 1.0))
    sampler = SobolSampler(config_space, 8)
    samples = sampler.generate()
    assert len(samples) == 8


def test_latin_hypercube_sampler_generation():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(UniformFloatHyperparameter("param1", 0.0, 1.0))
    sampler = LatinHypercubeSampler(config_space, 10)
    samples = sampler.generate()
    assert len(samples) == 10


def test_halton_sampler_generation():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(UniformFloatHyperparameter("param1", 0.0, 1.0))
    sampler = HaltonSampler(config_space, 10)
    samples = sampler.generate()
    assert len(samples) == 10
