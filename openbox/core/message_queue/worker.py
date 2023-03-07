# License: MIT

import time
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.utils.util_funcs import parse_result
from openbox.core.message_queue.worker_messager import WorkerMessager
from openbox.utils.history import Observation


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
            config, timeout, FAILED_PERF = msg

            # Start working
            # evaluate configuration on objective_function
            obj_args, obj_kwargs = (config,), dict()
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
                objectives, constraints, extra_info = parse_result(ret)
            else:
                objectives, constraints, extra_info = FAILED_PERF.copy(), None, None

            observation = Observation(
                config=config, objectives=objectives, constraints=constraints,
                trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
            )

            # Send result
            print("Worker: observation=%s. sending result." % str(observation))
            try:
                self.worker_messager.send_message(observation)
            except Exception as e:
                print("Worker send message error:", str(e))
                return
