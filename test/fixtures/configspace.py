import pytest
from openbox import space as sp
'''
记得换下种子，不然在一个space内了
'''
@pytest.fixture
def configspace_tiny() -> sp.Space:
    cs = sp.Space(seed=0)
    x1 = sp.Real("x1", -5, 10, default_value=0)
    x2 = sp.Real("x2", 0, 15, default_value=0)
    cs.add_variables([x1, x2])

    return cs
'''
添加其他不同的夹具
'''
@pytest.fixture
def configspace_cat()-> sp.Space:
    cs_1 = sp.Space(seed=1)
    name = 'cat_hp'
    choices = ['red', 'green', 'blue']
    default_value = 'red'
    meta = {'description': 'Test Meta Data'}
    weights = [0.2, 0.3, 0.5]

    # Create an instance of the Categorical variable
    categorical_var = sp.Categorical(name=name, choices=choices, default_value=default_value, meta=meta,
                                     weights=weights)
    #注意这里是add_variable不能是variables
    cs_1.add_variable(categorical_var)
    return cs_1

@pytest.fixture
def configspace_big()-> sp.Space:
    #100个连续案例
    cs = sp.Space(seed=2)
    for i in range(100):
        x1 = sp.Real("x{}".format(i), -i, 10, default_value=0)
        x2 = sp.Real("x{}".format(i+100), 0, 15, default_value=0)
        cs.add_variables([x1, x2])
    return cs

@pytest.fixture
def configspace_mid()-> sp.Space:
    #15个连续案例
    cs = sp.Space(seed=3)
    for i in range(15):
        x2 = sp.Real("x{}".format(i-200), 0, 15, default_value=0)
        x1 = sp.Real("x{}".format(i-100), -i, 10, default_value=0)
        cs.add_variables([x1, x2])
    return cs


@pytest.fixture
def configspace_2() -> sp.Space:
    cs = sp.Space(seed=0)
    x1 = sp.Real(name='x1', lower=0, upper=10, default_value=5)
    x2 = sp.Categorical(name='x2', choices=["car", "train", "plane"], default_value="train")
    cs.add_variables([x1, x2])

    return cs

@pytest.fixture
def configspace_huge() -> sp.Space:
    cs = sp.Space(seed=0)
    x1 = sp.Real("x1", -5, 10, default_value=0)
    x2 = sp.Real("x2", 0, 15, default_value=0)
    x3 = sp.Categorical(name='x3', choices=["car", "train", "plane"], default_value="train")
    x4 = sp.Int("x4", -5, 15, default_value=0)
    cs.add_variables([x1, x2, x3, x4])
    
    return cs