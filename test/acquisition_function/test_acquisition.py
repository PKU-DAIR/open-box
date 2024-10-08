import pytest
import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox.acquisition_function.acquisition import (
    AbstractAcquisitionFunction,
    EI,
    EIC,
    EIPS,
    LogEI,
    LPEI,
    PI,
    LCB,
    Uncertainty,
)
'''
加入其他函数的测试
'''
from openbox.acquisition_function.multi_objective_acquisition import (
    MaxvalueEntropySearch,
    MESMO,
    MESMOC,
    MESMOC2,
)   
class ConfigurationMock:
    def __init__(self, values=None):
        self.values = values
        self.configuration_space = ConfigurationSpace()

    def get_array(self):
        return np.array(self.values, dtype=np.float64)
    
class MockModel:
    '''
    其实axis的重点在于方向，而不是行和列。具体到各种用法而言也是如此。
    当axis=1时，如果是求平均，那么是从左到右横向求平均；如果是拼接，那么也是左右横向拼接；如果是drop，那么也是横向发生变化，体现为列的减少。
    mean_col_vector = np.mean(X, axis=1).reshape(-1, 1)
return mean_col_vector, mean_col_vector
    这里代码重复了？
    '''
    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))]).reshape((-1, 1)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))]
        ).reshape((-1, 1))
    
    def predict(self, X: np.ndarray):
        return self.predict_marginalized_over_instances(X)
    

class MockModelDual:
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 2)), np.array(
            [np.mean(X, axis=1).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 2))
    
class MockModelDual_zerovar:
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized_over_instances(self, X):
        '''
        这个乘法真的没有问题？不是引用？
        np.full(X.shape[0], 0.0)：创建一个与 X 的行数相同的全 0 数组。
        >>> b=[a]
        >>> b
        [array([1, 2])]
        >>> b*2
        [array([1, 2]), array([1, 2])]
        >>> c=b*2
        >>> c
        [array([1, 2]), array([1, 2])]
        >>> c[0][0]=2
        >>> c
        [array([2, 2]), array([2, 2])]
        >>>
        '''
        return np.array([np.mean(X, axis=1).reshape((1, -1))] * self.num_targets).reshape((-1, 2)), np.array(
            [np.full(X.shape[0], 0.0).reshape((1, -1))] * self.num_targets
        ).reshape((-1, 2))
    
class MockConstraintModel:
    '''
    np.full(m.shape, 0.1)：创建一个与 m 形状相同的数组，数组中的每个元素都为 0.1
    '''
    def predict_marginalized_over_instances(self, X):
        m = -np.abs(np.mean(X, axis=1)).reshape((-1, 1))
        v = np.full(m.shape, 0.1)
        return m, v
    
class MockModelWithNaN:
    def predict_marginalized_over_instances(self, X):
        return np.full(X.shape[0], np.nan).reshape((-1, 1)), np.full(X.shape[0], np.nan).reshape((-1, 1))
    
class MockModelWrongVShape:
    #错的v shape？
    def predict_marginalized_over_instances(self, X):
        m = np.mean(X, axis=1).reshape((-1, 2))
        v = np.var(X, axis=1).reshape((-1, 1)) 
        return m, v


@pytest.fixture
def model():
    return MockModel()

@pytest.fixture
def model_eips():
    return MockModelDual()

@pytest.fixture
def model_eips_zerovar():
    return MockModelDual_zerovar()

@pytest.fixture
def constraint_model():
    return MockConstraintModel()

@pytest.fixture
def model_with_nan():
    return MockModelWithNaN()

@pytest.fixture
def model_wrong_v_shape():
    return MockModelWrongVShape()

@pytest.fixture
def acquisition_function(model):
    return EI(model)



# --------------------------------------------------------------
# Test AbstractAcquisitionFunction
# --------------------------------------------------------------

def test_update_model_and_eta(model, acquisition_function):
    model = "abc"
    assert acquisition_function.eta is None
    acquisition_function.update(model=model, eta=0.1)
    assert acquisition_function.model == model
    assert acquisition_function.eta == 0.1


def test_update_with_kwargs(acquisition_function):
    acquisition_function.update(model="abc", eta=0.0, other="hi there:)")
    assert acquisition_function.model == "abc"

def test_call_with_nan(model_with_nan):
    ei = EI(model=model_with_nan)
    ei.update(eta=1.0)

    configurations = [ConfigurationMock([0.5])]
    acq = ei(configurations)

    assert np.all(acq == -np.finfo(np.float64).max)

def test_convert_false_with_1x1(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = np.array([0.5])
    acq = ei(configurations, convert=False)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.5998206141871228)

def test_notimplemented():
    with pytest.raises(TypeError):
        instance = AbstractAcquisitionFunction()
        instance([ConfigurationMock([0.5])])


# --------------------------------------------------------------
# Test EI
# --------------------------------------------------------------

def test_ei_init(acquisition_function):
    ei = acquisition_function
    assert ei.long_name == 'Expected Improvement'
    assert abs(ei.par) < 1e-5
    assert str(ei) == 'EI (Expected Improvement)'

def test_ei_eta_fail(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model)
    configurations = [ConfigurationMock([1.0])]
    with pytest.raises(ValueError):
        '''
        1. with pytest.raises(ValueError):
pytest.raises 是 pytest 提供的一个上下文管理器，用于检查某段代码是否抛出特定类型的异常。
ValueError 是期望的异常类型。
这段代码的含义是：在 with 语句的块内，pytest 期望执行的代码会引发 ValueError。如果代码没有抛出 ValueError，测试将会失败。
        '''
        ei(configurations)

def test_ei_1x1(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0])]
    acq = ei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.3989422804014327)


def test_ei_NxD(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.0)
    assert np.isclose(acq[1][0], 0.9002060113671223)
    assert np.isclose(acq[2][0], 0.3989422804014327)

def test_ei_zero_variance(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = np.array([ConfigurationMock([0.0])])
    acq = np.array(ei(configurations))
    assert np.isclose(acq[0][0], 0.0)

def test_ei_Nx1(model, acquisition_function):
    ei = acquisition_function
    ei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0001]),
        ConfigurationMock([1.0]),
        ConfigurationMock([2.0]),
    ]
    acq = ei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.9999)
    assert np.isclose(acq[1][0], 0.3989422804014327)
    assert np.isclose(acq[2][0], 0.19964122837424575)


# --------------------------------------------------------------
# Test EIC
# --------------------------------------------------------------

@pytest.fixture
def acq_eic_single(model, constraint_model):
    return EIC(model = model, constraint_models = [constraint_model])

@pytest.fixture
def acq_eic_multi(model, constraint_model):
    return EIC(model = model, constraint_models = [constraint_model, constraint_model])

def test_eic_init(acq_eic_single):
    assert acq_eic_single.long_name == 'Expected Constrained Improvement'

def test_eic_single_constraint(model, acq_eic_single):
    eic = acq_eic_single
    eic.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([0.5])]
    acq = eic(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.5656769359904809)

def test_eic_multi_constraint(model, acq_eic_multi):
    eic = acq_eic_multi
    eic.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = eic(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.25010115992223825)
    assert np.isclose(acq[1][0], 0.3506143218346858)
    assert np.isclose(acq[2][0], 0.39831801965532126)


# --------------------------------------------------------------
# Test EIPS
# --------------------------------------------------------------


@pytest.fixture
def acq_eips(model_eips):
    return EIPS(model_eips)


def test_eips_init(acq_eips):
    assert acq_eips.long_name == 'Expected Improvement per Second'

def test_eips_1xD(model_eips, acq_eips):
    eips = acq_eips
    eips.update(model=model_eips, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0]), ConfigurationMock([1.0, 1.0])]
    acq = eips(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.3989422804014327)


def test_eips_m_fail(model, acq_eips):
    eips = acq_eips
    eips.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0])]
    with pytest.raises(ValueError):
        eips(configurations)

def test_eips_v_fail(model_wrong_v_shape, acq_eips):
    eips = acq_eips
    eips.update(model=model_wrong_v_shape, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0]),ConfigurationMock([1.0, 1.0])]
    with pytest.raises(ValueError):
        eips(configurations)


def test_eips_1d_input(model_eips, acq_eips):
    eips = acq_eips
    eips.update(model=model_eips, eta=1.0)
    configurations = [ConfigurationMock([1.0]), ConfigurationMock([2.0])]
    acq = eips(configurations)

    assert acq.ndim == 2 
    assert acq.shape[1] == 1 

def test_eips_eta_fail(model_eips, acq_eips):
    eips = acq_eips
    eips.update(model=model_eips)
    configurations = [ConfigurationMock([1.0, 1.0]), ConfigurationMock([1.0, 1.0])]
    with pytest.raises(ValueError):
        eips(configurations)

def test_eips_1xD_zero_var(model_eips_zerovar, acq_eips):
    eips = acq_eips
    eips.update(model=model_eips_zerovar, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0]), ConfigurationMock([1.0, 1.0])]
    acq = eips(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0)

# --------------------------------------------------------------
# Test LogEI
# --------------------------------------------------------------

@pytest.fixture
def acq_logei(model):
    return LogEI(model)

def test_logei_init(acq_logei):
    assert acq_logei.long_name == 'Log Expected Improvement'


def test_logei_eta_fail(model, acq_logei):
    logei = acq_logei
    logei.update(model=model)
    configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    with pytest.raises(ValueError):
        logei(configurations)


def test_logei_1xD(model, acq_logei):
    logei = acq_logei
    logei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
    acq = logei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.6480973967332011)


def test_logei_NxD(model, acq_logei):
    logei = acq_logei
    logei.update(model=model, eta=1.0)
    configurations = [
        ConfigurationMock([0.1, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = logei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 1.6670107375002425)
    assert np.isclose(acq[1][0], 1.5570607606556273)
    assert np.isclose(acq[2][0], 0.6480973967332011)

def test_logei_1xD_zero_var(model, acq_logei):
    logei = acq_logei
    logei.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([0.0])]
    acq = logei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0)

# --------------------------------------------------------------
# Test LPEI
# --------------------------------------------------------------

@pytest.fixture
def acq_lpei(model):
    return LPEI(model)

def test_lpei_init(acq_lpei):
    assert acq_lpei.long_name == 'Expected Improvement with Local Penalizer'
    assert np.isclose(acq_lpei.estimate_L, 10.0)
    assert len(acq_lpei.batch_configs) == 0

def test_lpei_1x1(model, acq_lpei):
    lpei = acq_lpei
    lpei.update(model=model, eta=1.0)

    configurations = [ConfigurationMock([0.5])]
    batch_configs = [ConfigurationMock([0.6])]

    lpei.batch_configs = batch_configs

    acq = lpei(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], -0.09414608748817546)

def test_lpei_NxD(model, acq_lpei):
    lpei = acq_lpei
    lpei.update(model=model, eta=1.0)

    configurations = [
        ConfigurationMock([0.1, 0.0, 0.0]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    batch_configs = [
        ConfigurationMock([0.6, 0.6, 0.6]),
        ConfigurationMock([0.7, 0.7, 0.7])
    ]

    lpei.batch_configs = batch_configs

    acq = lpei(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.25386894076384675)
    assert np.isclose(acq[1][0], 0.21615949314476543)
    assert np.isclose(acq[2][0], -0.09196293793625696)


# --------------------------------------------------------------
# Test PI
# --------------------------------------------------------------

@pytest.fixture
def acq_pi(model):
    return PI(model)

def test_pi_init(acq_pi):
    assert acq_pi.long_name == 'Probability of Improvement'

def test_pi_eta_fail(model, acq_pi):
    pi = acq_pi
    pi.update(model=model)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    with pytest.raises(ValueError):
        pi(configurations)

def test_pi_1xD(model, acq_pi):
    pi = acq_pi
    pi.update(model=model, eta=1.0)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = pi(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.7602499389065233)


def test_pi_NxD(model, acq_pi):
    pi = acq_pi
    pi.update(model=model, eta=1.0)
    configurations = [  
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = pi(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 1.0)
    assert np.isclose(acq[1][0], 0.99778673707104)
    assert np.isclose(acq[2][0], 0.5)


# --------------------------------------------------------------
# Test LCB
# --------------------------------------------------------------


@pytest.fixture
def acq_lcb(model):
    return LCB(model)

def test_lcb_init(acq_lcb):
    assert acq_lcb.long_name == 'Lower Confidence Bound'
    assert acq_lcb.num_data == None

def test_lcb_numdata_fail(model, acq_lcb):
    lcb = acq_lcb
    lcb.update(model=model, eta=1.0, par=1)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    with pytest.raises(ValueError):
        lcb(configurations)

def test_lcb_1xD(model, acq_lcb):
    lcb = acq_lcb
    lcb.update(model=model, eta=1.0, par=1, num_data=3)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = lcb(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 1.315443985917585)
    lcb.update(model=model, eta=1.0, par=1, num_data=100)
    configurations = [ConfigurationMock([0.5, 0.5, 0.5])]
    acq = lcb(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 2.7107557771721433)


def test_lcb_1xD_no_improvement_vs_improvement(model, acq_lcb):
    lcb = acq_lcb
    lcb.update(model=model, par=1, num_data=1)
    configurations = [ConfigurationMock([0.001, 0.001])]
    acq = lcb(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 0.03623297)


def test_lcb_NxD(model, acq_lcb):
    lcb = acq_lcb
    lcb.update(model=model, eta=1.0, num_data=100)
    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]
    acq = lcb(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.045306943655446116)
    assert np.isclose(acq[1][0], 1.3358936353814157)
    assert np.isclose(acq[2][0], 3.5406943655446117)



# --------------------------------------------------------------
# Test Uncertainty
# --------------------------------------------------------------

@pytest.fixture
def acq_uncer(model):
    return Uncertainty(model)

def test_uncer_init(acq_uncer):
    assert acq_uncer.long_name == 'Uncertainty'
    assert acq_uncer.num_data == None

def test_uncer_numdata_fail(model, acq_uncer):
    uncertainty = acq_uncer
    uncertainty.update(model=model)
    configurations = [ConfigurationMock([0.5])]
    with pytest.raises(ValueError):
        uncertainty(configurations)


def test_uncertainty_1x1(model, acq_uncer):
    uncertainty = acq_uncer
    uncertainty.update(model=model, num_data=10)

    configurations = [ConfigurationMock([0.5])]

    acq = uncertainty(configurations)
    assert acq.shape == (1, 1)
    assert np.isclose(acq[0][0], 2.1459660262893476)

def test_uncertainty_NxD(model, acq_uncer):
    uncertainty = acq_uncer
    uncertainty.update(model=model, num_data=10)

    configurations = [
        ConfigurationMock([0.0001, 0.0001, 0.0001]),
        ConfigurationMock([0.1, 0.1, 0.1]),
        ConfigurationMock([1.0, 1.0, 1.0]),
    ]

    acq = uncertainty(configurations)
    assert acq.shape == (3, 1)
    assert np.isclose(acq[0][0], 0.03377508689746394)
    assert np.isclose(acq[1][0], 1.0680620276609596)
    assert np.isclose(acq[2][0], 3.377508689746394)

def test_uncertainty_with_nan(model_with_nan, acq_uncer):
    uncertainty = acq_uncer
    uncertainty.update(model=model_with_nan, num_data=10)
    configurations = [ConfigurationMock([0.5])]
    acq = uncertainty(configurations)

    assert np.all(acq == 0)




'''
import numpy as np

# 创建一个形状为 (2, 3, 4) 的三维数组
data = np.array([
    [[1, 2, 3, 4],   # 实验 1
     [5, 6, 7, 8],
     [9, 10, 11, 12]],

    [[13, 14, 15, 16],  # 实验 2
     [17, 18, 19, 20],
     [21, 22, 23, 24]]
])

print("原始数组:")
print(data)
mean_axis_0 = np.mean(data, axis=0)
print("沿 axis=0 计算的均值:")
print(mean_axis_0)
# 输出:
# [[ 7.  8.  9. 10.]
#  [11. 12. 13. 14.]
#  [15. 16. 17. 18.]]
mean_axis_1 = np.mean(data, axis=1)

print("沿 axis=1 计算的均值:")
print(mean_axis_1)
# 输出:
# [[ 5.  6.  7.  8.]
#  [19. 20. 21. 22.]]

mean_axis_neg1 = np.mean(data, axis=-1)
print("沿 axis=-1 计算的均值:")
print(mean_axis_neg1)
# 输出:
# [[ 2.5  6.5 10.5]
#  [14.5 18.5 22.5]]
总结
'''

'''
使用 np.expand_dims
当我们调用：
expanded = np.expand_dims(self.cell_lower_bounds, axis=(1, 2))
axis=(1, 2) 指定在第 1 和第 2 维的位置插入新维度。
变化步骤
第一次扩展 (axis=1):

原始数组形状为 (m,)，在第 1 维插入一个新维度后，形状变为 (1, m)：
python
复制代码
[[1, 2, 3]]  # 形状为 (1, 3)
第二次扩展 (axis=2):

在新数组的第 2 维插入新维度，形状变为 (1, m, 1)：
python
复制代码
[[[1],
  [2],
  [3]]]  # 形状为 (1, 3, 1)
总结
因此，经过两次扩展后，self.cell_lower_bounds 的形状最终变为 (1, m, 1)，表示有一个批次（第一维为 1），m 个目标（第二维为 m），
以及每个目标的单个值（第三维为 1）。这种形状使得该数组可以与其他三维数组（如 Y_samples）进行有效的广播和运算。
'''
