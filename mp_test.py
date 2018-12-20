
import pickle
import json
from multiprocessing import Queue, Process, cpu_count
import numpy as np
import rs_model

def add_helper(queue, arg1, arg2):  # the func called in child processes
    # ret = arg1 + arg2
    # r = arg2**2
    r = np.square(arg2)
    queue.put([arg1, r])


def multi_add():  # spawns child processes
    q = Queue()
    processes = []
    rets = []
    for i in range(0, 100):
        p = Process(target=add_helper, args=(q, i, i))
        processes.append(p)
        p.start()
    for p in processes:
        ret = q.get() # will block
        rets.append(ret)
    for p in processes:
        p.join()
    return rets


def pickle_t():
    f_name = 'rs_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)
    m = rs_model.RSModel(seed=3, config=config)
    p = pickle.dumps(m)
    # print(p)
    # b'\x80\x03c__main__\nFoo\nq\x00)\x81q\x01.'
    u = pickle.loads(p)


def get_rewards(queue, seed, config):
        m = rs_model.RSModel(seed=seed, config=config)
        reward, frames = m.evaluate_model()
        queue.put([seed, reward])


def main():

    import time
    import rs_model
    import json
    import pandas as pd
    from multiprocessing import Queue, Process

    # start timing
    start = time.time()

    f_name = 'rs_frostbite.json'

    our_seeds = []
    our_rewards = []

    with open(f_name) as f:
        config = json.load(f)

    # def get_rewards(queue, seed, config):
    #     m = rs_model.RSModel(seed=seed, config=config)
    #     reward, frames = m.evaluate_model()
    #     queue.put([seed, reward])

    q = Queue()
    processes = []
    our_seeds = []
    our_rewards = []
    for s in range(60):
        p = Process(target=get_rewards, args=(q, s, config))
        processes.append(p)
        p.start()
    for p in processes:
        seed, reward = q.get()  # will block
        our_seeds.append(seed)
        our_rewards.append(reward)
    for p in processes:
        p.join()

    elapsed = (time.time() - start)
    print("Non Sequential Time: " + str(round(elapsed)))

    df = pd.DataFrame({'seed': our_seeds, 'reward': our_rewards})
    print(df)


if __name__ == "__main__":
    main()
    # print("cpu count: " + str(cpu_count()))
    # a = multi_add()
    # print(a)
    # pickle_t()
