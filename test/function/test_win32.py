import psutil
from multiprocessing import Pool, Process, Pipe, freeze_support


def f(x, conn):
    print('process id', psutil.Process().pid)
    conn.send(x * x)
    return x * x


if __name__ == "__main__":
    freeze_support()
    # pool = Pool(processes=4)
    # r = pool.map(f, range(100))
    # pool.close()
    # pool.join()
    parent_conn, child_conn = Pipe(False)
    p1 = Process(target=f, args=(12, child_conn))
    p1.start()
    child_conn.close()

    p1.join(5)
    print(parent_conn.recv())
    parent_conn.close()

