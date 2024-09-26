import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from openbox.core.generic_advisor import Advisor
from openbox.utils.util_funcs import check_random_state
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_generic_advisor(configspace_tiny,configspace_cat,configspace_big,configspace_mid,history_single_obs):
    config_space = configspace_tiny
    #设置其他类型的config_space,需要修改fixtures下面的
    config_space_2 = configspace_cat
    #传入其他类型的参数,非连续和cat我不太知道，毕竟space.const的语法是啥？我看不了二进制，可能得去MIT源码看
    config_space_3 = configspace_big
    config_space_4 = configspace_mid

    advisor_cat = Advisor(config_space_2,early_stop=True,early_stop_kwargs={"min_iter":1})
    config=advisor_cat.get_suggestion()
    observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor_cat.update_observation(observation)

    advisor_cat.get_suggestion()
    #设置早停机制
    advisor_initial = Advisor(config_space)
    #传入early_stop_kwards可以提前终止，这个kwards中的min_iter是控制何时终止的，大于了就终止
    advisor = Advisor(config_space,early_stop=True)
    advisor_2 = Advisor(config_space,ref_point=[0, 0],num_objectives=2,num_constraints=1)
    advisor_3 = Advisor(config_space_3,num_objectives=2,ref_point=[0, 0],init_strategy='halton')
    advisor_3.get_suggestion()
    #advisor_3是ehvic
    advisor_4 = Advisor(config_space_4,num_objectives=5,ref_point=[0, 0, 0, 0, 0],init_strategy='random')
    advisor_4.get_suggestion()
    #我们直接修改advisor_4的acq_type，让那段代码能运行
    advisor_4.acq_type='ehvi'
    advisor_4.check_setup()
    advisor_4.acq_type='mesmo';advisor_4.surrogate_type='gp'
    advisor_4.check_setup()


    advisor_5 = Advisor(config_space_4,num_objectives=5,num_constraints=1,ref_point=[0, 0, 0, 0, 0],init_strategy='sobol')
    advisor_5.get_suggestion()
    #拿advisor_5撞set_up
    advisor_5.acq_type='ehvic'
    advisor_5.check_setup()
    advisor_5.acq_type='mesmoc';advisor_5.surrogate_type='gp'
    advisor_5.check_setup()
    advisor_5.acq_type='mesmoc';advisor_5.constraint_surrogate_type='gp'
    advisor_5.check_setup()
    advisor_6 = Advisor(config_space_4,num_constraints=1,init_strategy='latin_hypercube')
    advisor_6.get_suggestion()
    #这里建立usemo和parego
    # advisor_6.acq_type = 'parego'
    # advisor_6.setup_bo_basics()
    # advisor_6.acq_type = 'usemo'
    # advisor_6.setup_bo_basics()
    '''
    进行一些assert看是否与预期相同(to do)
    '''

    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.init_strategy == 'random_explore_first'
    assert advisor.rand_prob == 0.1
    assert advisor.optimization_strategy == 'bo'
    assert advisor.surrogate_type == 'gp'
    assert advisor.acq_type == 'ei'
    assert advisor.acq_optimizer_type == 'random_scipy'
    assert advisor.ref_point is None
    assert advisor.output_dir == 'logs'
    assert advisor.task_id == 'OpenBox'
    assert isinstance(advisor.rng, np.random.RandomState)
    config = advisor.get_suggestion(history=history_single_obs)
    assert config is not None

    observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert len(advisor.history) == 1

    configs = advisor.sample_random_configs(config_space, 5)
    assert len(configs) == 5

    history = advisor.get_history()
    assert history == advisor.history

    '''
    让initial configuration有值,注意传参不能跳过默认参数，要么就用关键字形式赋值
    '''
    initial_configurations = advisor.create_initial_design(advisor.init_strategy)
    new_advisor = Advisor(config_space,initial_configurations=initial_configurations)
    advisor.save_json("test/datas/test.json")

    advisor.load_json("test/datas/test.json")

