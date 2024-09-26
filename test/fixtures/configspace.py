import pytest
from openbox import space as sp

@pytest.fixture
def configspace_tiny() -> sp.Space:
    cs = sp.Space(seed=0)
    x1 = sp.Real("x1", -5, 10, default_value=0)
    x2 = sp.Real("x2", 0, 15, default_value=0)
    cs.add_variables([x1, x2])

    return cs