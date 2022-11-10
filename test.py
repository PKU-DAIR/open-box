from openbox import sp
space = sp.Space()
# 两个实数型（浮点型）变量
x1 = sp.Real('x1', -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])

# 整形变量
i = sp.Int("i", 0, 100)
# 类别型变量
kernel = sp.Categorical("kernel", ["rbf", "poly", "sigmoid"], default_value="rbf")

import numpy as np

# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return y


from openbox import Optimizer
import os

# Run
opt = Optimizer(
    branin, # 目标函数
    space, # 搜索空间
    num_constraints=0, # 没有约束条件
    num_objs=1, # 单目标值
    max_runs=50, # 优化目标函数50次
    surrogate_type='gp', # 用高斯过程
    time_limit_per_trial=30, # 最大时间预算30s
    task_id='test_03', # 区别不同优化过程
    # json_path=os.path.join(os.path.abspath("."), 'bo_history'),
)
history = opt.run()
