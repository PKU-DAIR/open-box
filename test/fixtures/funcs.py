import pytest
import numpy as np

@pytest.fixture
def func_brain():

    # Define Objective Function
    def branin(config):
        x1, x2 = config['x1'], config['x2']
        y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
        return {'objectives': [y]}
    
    return branin
