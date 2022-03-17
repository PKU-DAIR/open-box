import os
import sys
sys.path.append(os.getcwd())
from openbox.utils.limit import time_limit


# def test_func(*args, **kwargs):
#     import time
#     n = args[0]
#     time.sleep(n)
#     return n * n


def test_func(*args, **kwargs):
    import numpy as np
    # change mat_n: {10000, 1000, 100, 10}
    mat_n = 1000
    m = np.random.random((mat_n, mat_n))
    from sklearn.decomposition import KernelPCA

    for _ in range(1000):
        pca = KernelPCA()
        pca.fit_transform(m)
    return m * m


if __name__ == "__main__":
    # change the value of a:
    #     (1) a = (3)
    #     (2) a = (6)

    a = (3,)
    b = dict()
    res = time_limit(test_func, 5, a, b)
    # res = ps_time_limit(test, a, b, 1)
    print(res)
