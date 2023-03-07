# License: MIT

import time
import numpy as np
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.core.message_queue.worker_messager import WorkerMessager


class mqmfWorker(object):
    """
    message queue worker for multi-fidelity optimization
    """
    def __init__(self, objective_function, ip="127.0.0.1", port=13579, authkey=b'abc'):
        self.objective_function = objective_function
        self.worker_messager = WorkerMessager(ip, port, authkey=authkey)

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
            config, extra_conf, timeout, n_iteration, trial_id = msg

            # Start working
            start_time = time.time()
            ref_id = None
            early_stop = False

            # evaluate configuration on objective_function
            obj_args, obj_kwargs = (config, n_iteration, extra_conf), dict()
            result = run_obj_func(self.objective_function, obj_args, obj_kwargs, timeout)

            # parse result
            ret, timeout_status, traceback_msg, elapsed_time = (
                result['result'], result['timeout'], result['traceback'], result['elapsed_time'])
            if timeout_status:
                trial_state = TIMEOUT
            elif traceback_msg is not None:
                trial_state = FAILED
                print(f'Exception raised in objective function:\n{traceback_msg}\nconfig: {config}')
            else:
                trial_state = SUCCESS
            if trial_state == SUCCESS:
                if isinstance(ret, dict):
                    perf = ret['objective_value']
                    if perf is None:
                        perf = np.inf
                    ref_id = ret.get('ref_id', None)
                    early_stop = ret.get('early_stop', False)
                else:
                    perf = ret
            else:
                perf = np.inf

            time_taken = time.time() - start_time
            return_info = dict(loss=perf,
                               n_iteration=n_iteration,
                               ref_id=ref_id,
                               early_stop=early_stop,
                               trial_state=trial_state)
            observation = [return_info, time_taken, trial_id, config]

            # Send result
            print("Worker: perf=%f. time=%d. sending result." % (perf, int(time_taken)))
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                print("Worker send message error:", str(e))
                return
