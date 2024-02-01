import numpy as np
from openbox.utils.multi_objective.scalarization import get_chebyshev_scalarization


def test_get_chebyshev_scalarization():
    weights, Y = np.array([0.75, 0.25]), np.array([[1, 2], [2, 3], [3, 4]])

    # Test the scalarization function
    chebyshev_scalarization = get_chebyshev_scalarization(weights, Y)

    # Test if the scalarization function returns the correct shape
    assert chebyshev_scalarization(Y).shape == (len(Y),)

    # Test if the scalarization values are within the expected range
    assert all(0 <= val <= 1 for val in chebyshev_scalarization(Y))
