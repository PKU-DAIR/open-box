# License: MIT

import queue
from multiprocessing.managers import BaseManager
from openbox.utils.platform import get_platform


class MasterMessager(object):
    def __init__(self, ip="", port=13579, authkey=b'abc', max_send_len=100, max_rev_len=100):
        self.ip = ip
        self.port = port
        self.authkey = authkey
        self.max_sendqueue_length = max_send_len
        self.max_revqueue_length = max_rev_len
        self.masterQueue = None
        self.workerQueue = None
        self._masterQueue_for_win = None
        self._workerQueue_for_win = None
        self._init_master()

    def _init_master(self):
        if get_platform() == 'Windows':
            # [BUG FIX] Windows does not support lambda function in register
            QueueManager.register('get_master_queue', callable=self._get_master_queue_for_win)
            QueueManager.register('get_worker_queue', callable=self._get_worker_queue_for_win)
        else:  # Linux and OSX(MacOS)
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

    def _get_master_queue_for_win(self):
        if self._masterQueue_for_win is not None:
            # in worker
            return self._masterQueue_for_win
        # in master
        self._masterQueue_for_win = queue.Queue(maxsize=self.max_sendqueue_length)
        return self._masterQueue_for_win

    def _get_worker_queue_for_win(self):
        if self._workerQueue_for_win is not None:
            # in worker
            return self._workerQueue_for_win
        # in master
        self._workerQueue_for_win = queue.Queue(maxsize=self.max_revqueue_length)
        return self._workerQueue_for_win


class QueueManager(BaseManager):
    pass
