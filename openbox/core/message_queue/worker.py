# License: MIT

import time
import sys
import traceback
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import get_result
from openbox.core.message_queue.worker_messager import WorkerMessager
from openbox.core.base import Observation


class Worker(object):
    def __init__(self, objective_function, ip="127.0.0.1", port=13579, authkey=b'abc'):
        self.objective_function = objective_function
        self.worker_messager = WorkerMessager(ip, port, authkey)

    def run(self):
        while True:
            # Get config
            try:
                msg = self.worker_messager.receive_message()
            except Exception as e:
                print("Worker receive message error:", str(e))
                return
            if msg is None:
                # Wait for configs
                time.sleep(1)
                continue
            print("Worker: get config. start working.")
            config, time_limit_per_trial = msg

            # Start working
            trial_state = SUCCESS
            start_time = time.time()
            try:
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
                else:
                    objs, constraints = get_result(_result)
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                objs = None
                constraints = None

            elapsed_time = time.time() - start_time
            observation = Observation(
                config=config, objs=objs, constraints=constraints,
                trial_state=trial_state, elapsed_time=elapsed_time,
            )

            # Send result
            print("Worker: observation=%s. sending result." % str(observation))
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                print("Worker send message error:", str(e))
                return
