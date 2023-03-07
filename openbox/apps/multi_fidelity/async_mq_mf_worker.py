# License: MIT

import time
import numpy as np
from openbox import logger
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.core.message_queue.worker_messager import WorkerMessager


def no_time_limit_func(objective_function, timeout, args, kwargs):
    ret = objective_function(*args, **kwargs)
    return False, ret


class async_mqmfWorker(object):
    """
    async message queue worker for multi-fidelity optimization
    """
    def __init__(self, objective_function,
                 ip="127.0.0.1", port=13579, authkey=b'abc',
                 sleep_time=0.1,
                 no_time_limit=False):
        self.objective_function = objective_function
        self.worker_messager = WorkerMessager(ip, port, authkey=authkey)
        self.sleep_time = sleep_time

        if no_time_limit:
            self.time_limit = no_time_limit_func
        else:
            self.time_limit = run_obj_func

    def run(self):
        # tell master worker is ready
        init_observation = [None, None, None, None]
        try:
            self.worker_messager.send_message(init_observation)
        except Exception as e:
            logger.info("Worker send init message error: %s" % str(e))
            return

        while True:
            # Get config
            try:
                msg = self.worker_messager.receive_message()
            except Exception as e:
                logger.info("Worker receive message error: %s" % str(e))
                return
            if msg is None:
                # Wait for configs
                time.sleep(self.sleep_time)
                continue
            logger.info("Worker: get msg: %s. start working." % msg)
            config, extra_conf, timeout, n_iteration, trial_id = msg

            # Start working
            start_time = time.time()
            ref_id = None
            early_stop = False
            test_perf = None

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
                    test_perf = ret.get('test_perf', None)
                else:
                    perf = ret
            else:
                perf = np.inf

            time_taken = time.time() - start_time
            return_info = dict(loss=perf,
                               n_iteration=n_iteration,
                               ref_id=ref_id,
                               early_stop=early_stop,
                               trial_state=trial_state,
                               test_perf=test_perf)
            observation = [return_info, time_taken, trial_id, config]

            # Send result
            logger.info("Worker: perf=%f. time=%.2fs. sending result." % (perf, time_taken))
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                logger.info("Worker send message error: %s" % str(e))
                return
