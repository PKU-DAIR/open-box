# License: MIT

"""
Distributed Optimization Example

command line:
[Master]
python distributed_optimization.py --role master --n_workers 2 --parallel_strategy async --port 13579

[Worker]
python distributed_optimization.py --role worker --master_ip 127.0.0.1 --port 13579

"""

import argparse
import numpy as np
from openbox import sp, DistributedOptimizer, DistributedWorker

parser = argparse.ArgumentParser()
parser.add_argument('--role', type=str, choices=['master', 'worker'])
parser.add_argument('--n_workers', type=int)
parser.add_argument('--parallel_strategy', type=str, default='async', choices=['sync', 'async'])
parser.add_argument('--master_ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=13579)

# Parse args
args = parser.parse_args()
role = args.role
master_ip = args.master_ip
port = args.port
n_workers = args.n_workers

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objs': (y,)}


if __name__ == "__main__":
    if role == 'master':
        opt = DistributedOptimizer(
            branin,
            space,
            parallel_strategy='async',
            batch_size=n_workers,
            batch_strategy='default',
            num_objs=1,
            num_constraints=0,
            max_runs=50,
            # surrogate_type='gp',
            surrogate_type='auto',
            time_limit_per_trial=180,
            task_id='distributed_opt',
            ip="",
            port=port,
            authkey=b'abc',
        )
        history = opt.run()
        print(history)

    else:
        worker = DistributedWorker(branin, master_ip, port, authkey=b'abc')
        worker.run()
