from multiprocessing import Process, Queue
import os
import time

def f(q, flag):
    while True:
        for i in range(40):
            q.put([42, None, 'hello', i])
            time.sleep(0.1)
        if flag.empty():
            flag.put(True)


def g(q):
    while True:
        # if not q.empty():
        # print(q.get())

        print(q.get())
    # q.put([35])

if __name__ == '__main__':
    q = Queue()
    flag = Queue(maxsize = 1)
    p = Process(target=f, args=(q,flag))
    finished = False
    t = Process(target=g, args=(q,))
    p.start()
    t.start()
    process_id = p.pid
    # print(q.get())    # prints "[42, None, 'hello']"
    # q.put(['bla'])
    # if (not q.empty()):
    #     print(q.get())    # prints "[42, None, 'hello']"
    # else:
    #     print("ta vazia")

    # tmp = os.popen("ps -Af").read()

    # print(tmp)
    print(p.is_alive())
    print(process_id)
    # p.join()
    print("esperando")
    print(p.is_alive())

    # while str(process_id) in tmp[:]:
    while p.is_alive() and not finished:
        # print(flag.empty())
        if not flag.empty():
            finished = flag.get()
        # print "The process is running."

    print("to aqui tio")
    p.terminate()
    t.terminate()
    # t.join()
