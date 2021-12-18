#AUTHOR: Leo A. Bianchi - 2021

import multiprocessing as mp
import time as t
import numpy as np
import Wolff as W

def worker (r):
    C, err = W.C(r, 0.5)
    return [r, C, err]

if __name__ == '__main__':
    print('START')
    t1 = t.time()
    rs = np.arange(0, 16, 1)
    pool = mp.Pool(processes=6)
    res = pool.map(worker, rs)
    t2 = t.time()
    print('END, time taken = '+str((t2-t1)/60)+' minutes')
    np.savetxt('Cs_05.txt', res)
