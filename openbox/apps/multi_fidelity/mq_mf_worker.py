# License: MIT

import time
import sys
import traceback
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
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
            config, extra_conf, time_limit_per_trial, n_iteration, trial_id = msg

            # Start working
            start_time = time.time()
            trial_state = SUCCESS
            ref_id = None
            early_stop = False
            try:
                args, kwargs = (config, n_iteration, extra_conf), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
                else:
                    if _result is None:
                        perf = MAXINT
                    elif isinstance(_result, dict):
                        perf = _result['objective_value']
                        if perf is None:
                            perf = MAXINT
                        ref_id = _result.get('ref_id', None)
                        early_stop = _result.get('early_stop', False)
                    else:
                        perf = _result
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                perf = MAXINT

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
