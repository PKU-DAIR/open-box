# License: MIT

import sys
import time
import traceback
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.core.message_queue.worker_messager import WorkerMessager


def no_time_limit_func(objective_function, time_limit_per_trial, args, kwargs):
    ret = objective_function(*args, **kwargs)
    return False, ret


class async_mqmfWorker(object):
    """
    async message queue worker for multi-fidelity optimization
    """
    def __init__(self, objective_function,
                 ip="127.0.0.1", port=13579, authkey=b'abc',
                 sleep_time=0.1,
                 no_time_limit=False,
                 logger=None):
        self.objective_function = objective_function
        self.worker_messager = WorkerMessager(ip, port, authkey=authkey)
        self.sleep_time = sleep_time

        if no_time_limit:
            self.time_limit = no_time_limit_func
        else:
            self.time_limit = time_limit

        if logger is not None:
            self.logging = logger.info
        else:
            self.logging = print

    def run(self):
        # tell master worker is ready
        init_observation = [None, None, None, None]
        try:
            self.worker_messager.send_message(init_observation)
        except Exception as e:
            self.logging("Worker send init message error: %s" % str(e))
            return

        while True:
            # Get config
            try:
                msg = self.worker_messager.receive_message()
            except Exception as e:
                self.logging("Worker receive message error: %s" % str(e))
                return
            if msg is None:
                # Wait for configs
                time.sleep(self.sleep_time)
                continue
            self.logging("Worker: get msg: %s. start working." % msg)
            config, extra_conf, time_limit_per_trial, n_iteration, trial_id = msg

            # Start working
            start_time = time.time()
            trial_state = SUCCESS
            ref_id = None
            early_stop = False
            test_perf = None
            try:
                args, kwargs = (config, n_iteration, extra_conf), dict()
                timeout_status, _result = self.time_limit(self.objective_function,
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
                        test_perf = _result.get('test_perf', None)
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
                               trial_state=trial_state,
                               test_perf=test_perf)
            observation = [return_info, time_taken, trial_id, config]

            # Send result
            self.logging("Worker: perf=%f. time=%.2fs. sending result." % (perf, time_taken))
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                self.logging("Worker send message error: %s" % str(e))
                return
