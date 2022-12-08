# License: MIT
import random

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation
from openbox.core.ea.differential_ea_advisor import DifferentialEAAdvisor
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.core.ea.adaptive_ea_advisor import AdaptiveEAAdvisor
from openbox.core.ea.cmaes_ea_advisor import CMAESEAAdvisor
from openbox.core.ea.nsga2_ea_advisor import NSGA2EAdvisor

from openbox.benchmark.objective_functions.synthetic import Rosenbrock
from openbox.benchmark.objective_functions.synthetic import BaseTestProblem
from openbox.benchmark.objective_functions.synthetic import UniformFloatHyperparameter
from openbox.benchmark.objective_functions.synthetic import Configuration, ConfigurationSpace

# Define Objective Function
from openbox.core.ea.saea_advisor import SAEAAdvisor
from openbox.utils.config_space import convert_configurations_to_array

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range


class TestFunc(BaseTestProblem):
    r"""
    https://www.bbsmax.com/A/kjdwWePqdN/

    #69 min f(x,y)=-(x*sin(9*pi*y)+y*cos(25*pi*x)+20);
    #   -10<=x<=10, -10<=y<=10;
    #   使其满足非线性不等式 x^2+y^2<=9^2;
    #   因为 D的面积很大 ,所以sin (9 πy) 与cos (25 πx) 分别在不同方向
    #   上高频振荡。 函数的深谷密集在圆 x^2+y^2<=9^2上的4个点
    #   (9/sqrt(2),,9/sqrt(2)) , (9/sqrt(2) , - 9/sqrt(2)) ,
    #   (-9/sqrt(2) ,9/sqrt(2)) , ( - 9/sqrt(2),- 9/sqrt(2)的附近。
    #   求解这个函数的优化问题不仅传统的算法无能为力 ,而且即使采用最新的
    #   演化算法(遗传算法或演化策略等)也很难求解。
    #   采用郭涛算法获得的精确到小数点后 14 位的最优解是
    #   minf(x,y)=f(- 6.440025882216888 , - 6.27797201362437)
    #            = - 32.71788780688353 ,
    #   这个最优解在10次运算中只得到了1～3次。在此应注明的是,该问题的真
    #   正最优解是未知的,认为它可能就是最优解。
    #   摘自：郭涛算法及应用  李艳　康卓　刘溥 (武汉大学计算中心)
    """

    def __init__(self, noise_std = 0, random_state = None):
        self.ref_point = [0.0, 0.0]

        params = {'x1': (-10.0, 10.0),
                  'x2': (-10.0, 10.0)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         num_objectives = 1, num_constraints = 1,
                         random_state = random_state)

    def _evaluate(self, X):
        result = dict()
        obj = -(X[..., 0] * np.sin(9 * np.pi * X[..., 1]) + X[..., 1] * np.cos(25 * np.pi * X[..., 0]) + 20)
        result['objectives'] = (obj,)

        c = X[..., 0] ** 2 + X[..., 1] ** 2 - 9 ** 2
        result['constraints'] = (c,)

        return result


CNUM = 0
function = Rosenbrock()
function.constrained = True
if function.constrained is True:
    CNUM = 1
else:
    CNUM = 0
space = function.config_space

# Run
if __name__ == "__main__":
    advisors = [RegularizedEAAdvisor(
        config_space = space,
        task_id = 'OpenBox',
        num_objectives = 1,
        num_constraints = CNUM
    ), SAEAAdvisor(
        config_space = space,
        task_id = 'OpenBox',
        ea = RegularizedEAAdvisor,
        num_objectives = 1,
        num_constraints = CNUM
    )]

    res = function(space.sample_configuration())
    dim = len(res['objectives'])

    axes = None
    histories = []

    MAX_RUNS = 100
    for advisor in advisors:
        print("Now running" + str(advisor.__class__))
        m = MAX_RUNS

        for i in trange(m):
            # ask
            config = advisor.get_suggestion()
            # evaluate
            ret = function(config)
            # tell
            observation = Observation(config = config, objectives = ret['objectives'],
                                      constraints = ret['constraints'] if 'constraints' in ret.keys() else None)
            advisor.update_observation(observation)
            # print(observation.objectives)

            if trange == range:
                print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))

        history = advisor.get_history()
        # print('configs', len(history.get_all_configs()))
        histories.append(history.get_incumbents())

        if dim == 1:
            axes = history.plot_convergence(ax = axes)
        elif dim == 2:
            inc = history.get_incumbents()
            inc.sort(key = lambda x: x[1][0])
            plt.plot([x[1][0] for x in inc], [x[1][1] for x in inc], label = advisor.__class__.__name__)

    if dim <= 2:
        plt.legend()
        plt.show()

    print('--------------------OPTIMIZATION RESULTS--------------------')

    for i, h in enumerate(histories):
        print(advisors[i].__class__.__name__)
        print(h)

    print('--------------------SAEA CORRECTNESS CHECK--------------------')

    saea = advisors[1]
    gp = saea.objective_surrogates[0]

    # print(saea.lastX)
    print('--------------------LAST GP TRAINING DATA--------------------')

    print('total {} data'.format(len(saea.lastX)))

    print('randomly print 10 of them: (X, Y, f(X))')

    rand_data = list(zip(saea.lastX,
                         saea.lastY,
                         [function(Configuration(space, vector = saea.lastX[i])) for i in range(saea.lastX.shape[0])]))
    random.shuffle(rand_data)

    for i, x in enumerate(rand_data):
        if i >= 10:
            break
        print(x)

    print('--------------------GP PREDICTION CORRECTNESS--------------------')
    print('randomly sample 10 configs, get their gp-prediction and true value: (X, gp(X), f(X))')

    for i in range(10):
        config = space.sample_configuration()

        pred = gp.predict(convert_configurations_to_array([config]))
        target = function(config)
        print(config.get_array(), pred, target)
        # print(convert_configurations_to_array([config]), pred, target)
