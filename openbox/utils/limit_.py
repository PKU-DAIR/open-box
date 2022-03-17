# License: MIT

import sys
import dill
import time
import psutil
from collections import namedtuple
from multiprocessing import Process, freeze_support, Pipe


class SignalException(Exception):
    pass


class TimeoutException(Exception):
    pass


class OutOfMemoryLimitException(Exception):
    pass


def get_platform():
    platforms = {
        'linux': 'Linux',
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OSX',
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        raise ValueError('Unsupported platform - %s.' % sys.platform)
    return platforms[sys.platform]


_platform = get_platform()
if _platform not in ['Windows']:
    import resource
Returns = namedtuple('return_values', ['status', 'result'])


def wrapper(*args, **kwargs):
    # Parse args.
    _func, _conn, _time_limit, _mem_limit, args = args[0], args[1], args[2], args[3], args[4:]
    _func = dill.loads(_func)
    result = (False, None)

    if _platform in ['Linux']:
        import signal

        def handler(signum, frame):
            if signum == signal.SIGALRM:
                raise TimeoutException
            else:
                raise SignalException

        # # Limit the memory usage.
        # if _mem_limit is not None:
        #     # Transform megabyte to byte
        #     mem_in_b = _mem_limit * 1024 * 1024
        #
        #     # Set the maximum size (in bytes) of address space.
        #     resource.setrlimit(resource.RLIMIT_AS, (mem_in_b, mem_in_b))

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(_time_limit)
    try:
        # print('start to call the function.')
        result = (True, _func(*args, **kwargs))
        # print('calling ends.')
    except TimeoutException:
        result = (False, TimeoutException)
    except MemoryError:
        result = (False, OutOfMemoryLimitException)
    except SignalException:
        result = (False, SignalException)

    finally:
        try:
            _conn.send(result)
            _conn.close()
        except:
            pass
        finally:
            p = psutil.Process()
            for child in p.children(recursive=True):
                child.kill()


def clean_processes(proc):
    if psutil.pid_exists(proc.pid):
        p = psutil.Process(proc.pid)
        p.terminate()
        for child in p.children(recursive=True):
            child.kill()


def limit_function(func, wall_clock_time, mem_usage_limit, *args, **kwargs):
    """
    :param func: the objective function to call.
    :param wall_clock_time: seconds.
    :param mem_usage_limit: megabytes.
    :param args:
    :param kwargs:
    :return:

    More Info about Memory Management [https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info]:
        rss: aka “Resident Set Size”, this is the non-swapped physical memory a process has used.
             On UNIX it matches “top“‘s RES column).
             On Windows this is an alias for wset field and it matches “Mem Usage” column of taskmgr.exe.
        vms: aka “Virtual Memory Size”, this is the total amount of virtual memory used by the process.
             On UNIX it matches “top“‘s VIRT column.
             On Windows this is an alias for pagefile field and it matches “Mem Usage” “VM Size” column of taskmgr.exe.
    """

    _debug_mode = False
    if 'debug' in kwargs:
        _debug_mode = kwargs['debug']

    # Deal with special case in Bayesian optimization.
    if len(args) == 0 and 'args' in kwargs:
        args = kwargs['args']
        kwargs = kwargs['kwargs']

    if _platform == 'Windows':
        freeze_support()

    parent_conn, child_conn = Pipe(False)
    func = dill.dumps(func)
    args = [func] + [child_conn] + [wall_clock_time] + [mem_usage_limit] + list(args)

    p = Process(target=wrapper, args=tuple(args), kwargs=kwargs)
    p.start()

    # Special case on windows.
    if _platform in ['Windows', 'OSX', 'Linux']:
        p_id = p.pid
        exceed_mem_limit = False
        start_time = time.time()
        while time.time() <= start_time + wall_clock_time:
            if not psutil.pid_exists(p_id) or psutil.Process(p_id).status() == 'zombie':
                break
            rss_used = psutil.Process(p_id).memory_info().rss / 1024 / 1024
            vms_used = psutil.Process(p_id).memory_info().vms / 1024 / 1024
            if _debug_mode:
                print(psutil.Process(p_id).memory_info())
                print('mem[rss]_used', rss_used)
                print('mem[vms]_used', vms_used)
            threshold = rss_used if _platform in ['OSX', 'Linux', 'Windows'] else vms_used
            if threshold > mem_usage_limit:
                exceed_mem_limit = True
                break
            time.sleep(1.)

        if exceed_mem_limit:
            clean_processes(p)
            return Returns(status=False, result=OutOfMemoryLimitException)
        if p.is_alive():
            clean_processes(p)
            return Returns(status=False, result=TimeoutException)
        result = parent_conn.recv()
        parent_conn.close()
        return result
    else:
        p.join(wall_clock_time)
        if p.is_alive():
            clean_processes(p)
            return Returns(status=False, result=TimeoutException)
        result = parent_conn.recv()
        parent_conn.close()
        return result


"""
==============
for UBUNTU,
pmem(rss=18956288, vms=105123840, shared=2846720, text=2797568, lib=0, data=16629760, dirty=0)
mem[rss]_used 17.67578125
mem[vms]_used 100.25390625
matrix size in megabytes 8.00006103515625
matrix size in megabytes 80.00006103515625
pmem(rss=82104320, vms=767774720, shared=28819456, text=2797568, lib=0, data=384090112, dirty=0)
mem[rss]_used 78.30078125
mem[vms]_used 732.20703125
(True, 12)

==============
for MACOSX,
pmem(rss=1470464, vms=4380520448, pfaults=449, pageins=0)
mem[rss]_used 1.2265625
mem[vms]_used 4177.58984375
matrix size in megabytes 8.00006103515625
matrix size in megabytes 80.00006103515625
pmem(rss=151318528, vms=4864417792, pfaults=39045, pageins=0)
mem[rss]_used 144.30859375
mem[vms]_used 4639.0703125
pmem(rss=66502656, vms=4872224768, pfaults=41080, pageins=0)
mem[rss]_used 63.421875
mem[vms]_used 4646.515625

===============
for WINDOWS,
pmem(rss=2719744, vms=1413120, num_page_faults=712, peak_wset=2723840, wset=2719744, peak_paged_pool=36040, paged_pool=36008, peak_nonpaged_pool=5160, nonpaged_pool=5160, pagefile=1413120, peak_pagefile=1413120, private=1413120)
mem[rss]_used 2.578125
mem[vms]_used 1.34765625
matrix size in megabytes 8.00006103515625
matrix size in megabytes 80.00006103515625
pmem(rss=61349888, vms=51875840, num_page_faults=40509, peak_wset=128303104, wset=61349888, peak_paged_pool=548648, paged_pool=548648, peak_nonpaged_pool=46080, nonpaged_pool=46080, pagefile=51875840, peak_pagefile=121303040, private=51875840)
mem[rss]_used 58.5078125
mem[vms]_used 49.47265625
pmem(rss=72642560, vms=93351936, num_page_faults=43329, peak_wset=128303104, wset=72642560, peak_paged_pool=582760, paged_pool=582760, peak_nonpaged_pool=54512, nonpaged_pool=54512, pagefile=93351936, peak_pagefile=121303040, private=93351936)
mem[rss]_used 69.27734375
mem[vms]_used 89.02734375


"""
