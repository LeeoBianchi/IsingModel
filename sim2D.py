#AUTHOR: Leo A. Bianchi - 2021

import multiprocessing as mp
import time as t
import numpy as np
import Wolff as W

def worker (TovJ):
    out = W.simulation_2D_1T(TovJ)
    return out

if __name__ == '__main__':
    print('START')
    t1 = t.time()
    TovJs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    pool = mp.Pool(processes=6)
    res = pool.map(worker, TovJs)
    t2 = t.time()
    print('END, time taken = '+str((t2-t1)/60)+' minutes')
    np.savetxt('sim2D_test1.txt', res)
