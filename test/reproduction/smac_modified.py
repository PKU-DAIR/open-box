import time as time_pkg
from smac.runhistory.runhistory import *


class RunHistory_modified(RunHistory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_start_time = time_pkg.time()
        self.global_trial_counter = 0
        self.config_list = []
        self.perf_list = []
        self.time_list = []

    def add(self, config: Configuration, cost: float, time: float,
            status: StatusType, instance_id: str = None,
            seed: int = None,
            additional_info: dict = None,
            origin: DataOrigin = DataOrigin.INTERNAL):
        # save record
        global_time = time_pkg.time() - self.global_start_time
        self.global_trial_counter += 1
        self.config_list.append(config)
        self.perf_list.append(cost)
        self.time_list.append(global_time)

        print('smac add record', self.global_trial_counter, config, cost, global_time, flush=True)
        # super class add
        super().add(config, cost, time, status, instance_id, seed, additional_info, origin)
