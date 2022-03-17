import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter


def branin_currin(config):
    dic = config.get_dictionary()
    x1 = dic.get('x1')
    x2 = dic.get('x2')
    px1 = 15*x1 - 5
    px2 = 15*x2
    res = dict()

    f1 = (px2 - 5.1/(4*np.pi**2) * px1**2 + 5/np.pi * px1 - 6)**2 + 10 * (1 - 1/(8*np.pi)) * np.cos(px1) + 10
    f2 = (1 - np.exp(-1/(2*x2))) * (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4*x1 + 20)
    res['objs'] = [f1, f2]
    res['constraints'] = []
    return res

bc_params = {
    'float': {
        'x1': (0, 1, 0.5),
        'x2': (0, 1, 0.5)
    }
}
bc_cs = ConfigurationSpace()
bc_cs.add_hyperparameters([UniformFloatHyperparameter(e, *bc_params['float'][e]) for e in bc_params['float']])
bc_max_hv = 59.36011874867746
bc_ref_point = [18., 6.]


bo = SMBO(branin_currin, bc_cs,
          task_id='parego',
          num_objs=2,
          acq_type='parego',
          ref_point=bc_ref_point,
          max_runs=100)
bo.run()

hvs = bo.get_history().hv_data
log_hv_diff = np.log10(bc_max_hv - np.asarray(hvs))
