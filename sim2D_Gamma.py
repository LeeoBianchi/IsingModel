#AUTHOR: Leo A. Bianchi - 2021

import multiprocessing as mp
import time as t
import numpy as np
import Wolff as W

def worker (TovJ):
    out8 = W.simulation_Gamma_1T(TovJ, L=8)[1] #I pick only the Gamma output
    out16 = W.simulation_Gamma_1T(TovJ, L=16)[1]
    out32 = W.simulation_Gamma_1T(TovJ, L=32)[1]
    return [out8, out16, out32]

if __name__ == '__main__':
    print('START')
    t1 = t.time()
    TovJs = np.linspace(2.24, 2.3, 48)
    pool = mp.Pool(processes=6)
    res = pool.map(worker, TovJs)
    t2 = t.time()
    print('END, time taken = '+str((t2-t1)/60)+' minutes')
    np.savetxt('sim2D_Gamma_zoom.txt', res)
