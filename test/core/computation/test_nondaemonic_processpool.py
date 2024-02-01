import multiprocessing
from openbox.core.computation.nondaemonic_processpool import NoDaemonProcess, NoDaemonContext, ProcessPool


def test_no_daemon_process_daemon_property():
    process = NoDaemonProcess()
    process.daemon = True
    assert process.daemon is False

    assert issubclass(NoDaemonContext.Process, multiprocessing.Process)


def test_process_pool_initialization():
    pool = ProcessPool()
    assert isinstance(pool, multiprocessing.pool.Pool)

