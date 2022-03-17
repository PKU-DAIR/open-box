# License: MIT

from multiprocessing.managers import BaseManager


class WorkerMessager(object):
    def __init__(self, ip="127.0.0.1", port=13579, authkey=b'abc'):
        self.ip = ip
        self.port = port
        self.authkey = authkey
        self.masterQueue = None
        self.workerQueue = None
        self._init_worker()

    def _init_worker(self):
        QueueManager.register('get_master_queue')
        QueueManager.register('get_worker_queue')
        manager = QueueManager(address=(self.ip, self.port), authkey=self.authkey)
        manager.connect()
        self.masterQueue = manager.get_master_queue()
        self.workerQueue = manager.get_worker_queue()

    def send_message(self, message):
        self.workerQueue.put(message)

    def receive_message(self):
        if self.masterQueue.empty() is True:
            return None
        message = self.masterQueue.get()
        return message


class QueueManager(BaseManager):
    pass
