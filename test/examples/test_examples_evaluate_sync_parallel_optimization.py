# License: MIT
import numpy as np
import matplotlib.pyplot as plt
from openbox import ParallelOptimizer, space as sp
from openbox.utils.constants import SUCCESS
from unittest.mock import Mock
from openbox.core.computation.parallel_process import ParallelEvaluation
from openbox.optimizer.parallel_smbo import wrapper
from openbox.core.computation.nondaemonic_processpool import ProcessPool
'''
这个也用来测试pSMBO 测试并没有发现wrapper被执行，暂时不知道原因
'''
# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objectives': [y]}

'''
尝试，但是wrapper依然没有执行
    # 使用 Mock 作为回调函数，检查是否调用
    mock_callback = Mock()
    proc = ProcessPool(4)
    _config = space.sample_configuration()
    _param = [branin, _config, None, [np.inf]]
    result = proc.apply_async(wrapper, (_param,), callback=mock_callback)

    # 等待任务完成
    result.wait()

    # 检查回调是否被调用
    mock_callback.assert_called_once()
'''

'''
直接调用wrapper
'''
def test_examples_evaluate_sync_parallel_optimization():
    max_runs = 20

    # Define Search Space
    space = sp.Space()
    x1 = sp.Real("x1", -5, 10, default_value=0)
    x2 = sp.Real("x2", 0, 15, default_value=0)
    space.add_variables([x1, x2])
    _config = space.sample_configuration()
    wrapper( (branin, _config, None, [np.inf]) )

    # Parallel Evaluation on Local Machine
    opt = ParallelOptimizer(
        branin,
        space,
        parallel_strategy='sync',
        batch_size=4,
        batch_strategy='default',
        num_objectives=1,
        num_constraints=0,
        max_runs=max_runs,
        # surrogate_type='gp',
        surrogate_type='auto',
        task_id='parallel_sync',
        logging_dir='logs/pytest/',
    )
    history = opt.run()
    # 测试更多并行方法
    opt_syn_ea = ParallelOptimizer(
    branin,
    space,
    parallel_strategy='sync',
    batch_size=4,
    batch_strategy='default',
    sample_strategy='ea',
    num_objectives=1,
    num_constraints=0,
    max_runs=max_runs,
    # surrogate_type='gp',
    surrogate_type='auto',
    task_id='parallel_sync',
    logging_dir='logs/pytest/',
)
    history = opt_syn_ea.run()

    opt_asyn_ea = ParallelOptimizer(
    branin,
    space,
    parallel_strategy='async',
    batch_size=4,
    batch_strategy='default',
    sample_strategy='ea',
    num_objectives=1,
    num_constraints=0,
    max_runs=max_runs,
    # surrogate_type='gp',
    surrogate_type='auto',
    task_id='parallel_sync',
    logging_dir='logs/pytest/',
)
    history = opt_asyn_ea.run()


    opt_asyn_default = ParallelOptimizer(
    branin,
    space,
    parallel_strategy='async',
    batch_size=4,
    batch_strategy='default',
    num_objectives=1,
    num_constraints=0,
    max_runs=max_runs,
    # surrogate_type='gp',
    surrogate_type='auto',
    task_id='parallel_sync',
    logging_dir='logs/pytest/',
)
    history = opt_asyn_default.run()
    opt_asyn_default.async_iterate()

    print(history)

    history.plot_convergence(true_minimum=0.397887)
    # plt.show()
    plt.savefig('logs/pytest/sync_parallel_convergence.png')
    plt.close()

    assert history.trial_states[:max_runs].count(SUCCESS) == max_runs
