from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])
    # q.put([35])

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    # q.put(['bla'])
    if (not q.empty()):
        print(q.get())    # prints "[42, None, 'hello']"
    else:
        print("ta vazia")
    p.join()
