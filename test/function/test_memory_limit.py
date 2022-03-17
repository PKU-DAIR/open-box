import os
import sys
sys.path.append(os.getcwd())
from openbox.utils.limit_ import limit_function


def test_func(*args, **kwargs):
    import numpy as np
    import sys
    print('matrix size in megabytes', sys.getsizeof([10000] * 1024 * 1024) / 1024 / 1024)
    m = [10000] * 1024 * 1024 * 10
    print('matrix size in megabytes', sys.getsizeof(m) / 1024 / 1024)
    mat_n = 100
    m = np.random.random((mat_n, mat_n))
    from sklearn.decomposition import KernelPCA

    for _ in range(1000):
        pca = KernelPCA()
        pca.fit_transform(m)

    return 12


if __name__ == "__main__":
    a = (3,)
    b = dict()
    res = limit_function(test_func, 20, 120, a, b, debug=True)
    print(res)
