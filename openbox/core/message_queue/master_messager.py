# License: MIT

import queue
from multiprocessing.managers import BaseManager


class MasterMessager(object):
    def __init__(self, ip="", port=13579, authkey=b'abc', max_send_len=100, max_rev_len=100):
        self.ip = ip
        self.port = port
        self.authkey = authkey
        self.max_sendqueue_length = max_send_len
        self.max_revqueue_length = max_rev_len
        self.masterQueue = None
        self.workerQueue = None
        self._init_master()

    def _init_master(self):
        _masterQueue = queue.Queue(maxsize=self.max_sendqueue_length)
        _workerQueue = queue.Queue(maxsize=self.max_revqueue_length)
        QueueManager.register('get_master_queue', callable=lambda: _masterQueue)
        QueueManager.register('get_worker_queue', callable=lambda: _workerQueue)
        manager = QueueManager(address=(self.ip, self.port), authkey=self.authkey)
        manager.start()
        self.masterQueue = manager.get_master_queue()
        self.workerQueue = manager.get_worker_queue()

    def send_message(self, message):
        self.masterQueue.put(message)

    def receive_message(self):
        if self.workerQueue.empty() is True:
            return None
        message = self.workerQueue.get()
        return message


class QueueManager(BaseManager):
    pass
