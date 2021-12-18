#AUTHOR: Leo A. Bianchi - 2021

import numpy as np
import os
import time as t
import numpy.random as rnd
import scipy.constants as const
import multiprocessing as mp
from matplotlib import pyplot as plt

kB = 1
L = 16
N_mean = 1000
n_step = 1  #we take measurments every n_step configs in the chain
nbin = 20
n_Temp = 12

def p (TovJ):
    return 1 - np.exp(- 2/(TovJ*kB))
    
    
def assign (TovJ):
    x = np.random.random()
    return x < p(TovJ)

#impplements the periodic boundary conditions
def PBC (ind, L=L):
    if ind == L:
        return 0 #PBC: L -> 0
    else:
        if ind == -1:
            return L-1 #PBC: -1 -> L-1
        else:
            return ind

#returns an array with the indexes (tuples) of the neighbors of the site, we only consider the 1D and 2D cases
def get_neighbors (site_index, L=L):
    if type(site_index) == tuple : #2D-case
        aux = []
        aux.append((PBC(site_index[0] + 1, L=L), site_index[1]))
        aux.append((PBC(site_index[0] - 1, L=L), site_index[1]))
        aux.append((site_index[0], PBC(site_index[1] + 1, L=L)))
        aux.append((site_index[0], PBC(site_index[1] - 1, L=L)))
        return aux #list of tuples
    else:  #1D-case
        aux = []
        aux.append(PBC(site_index + 1, L=L))
        aux.append(PBC(site_index - 1, L=L))
        return aux #list of ints

# 1) We randomly pick the first site
def step_1 (Ising):
    if len(np.shape(Ising)) == 1:
        site_index = rnd.randint(0, len(Ising))
        return site_index #int
    if len(np.shape(Ising)) == 2:
        site_index = []
        site_index.append(rnd.randint(0, np.alen(Ising[:,0])))
        site_index.append(rnd.randint(0, np.alen(Ising[0,:])))
        return tuple(site_index) #tuple

# 2) We flip the spin in site
def step_2 (site_index, Ising):
    Ising[site_index] = -Ising[site_index]
    return Ising

# 3) We consider the neighbors
def step_3 (site_index, Ising, TovJ):
    for neig_index in get_neighbors(site_index, L=len(Ising)): #we check both the neighbors
        if Ising[site_index] != Ising[neig_index]: #if the neighbor spin is antialigned
            if assign(TovJ): #I label the neighbor as the next site with prob p
                Ising = step_2(neig_index, Ising) #and go back to step 2
                Ising = step_3(neig_index, Ising, TovJ) #then step 3 again, ricursion
    return Ising

# Does N_eq updates to remove dependence of config on N
def equilibrate (Ising, TovJ, N_eq):
    for n in range(N_eq):
        site_index = step_1(Ising)
        Ising = step_2(site_index, Ising)
        Ising = step_3(site_index, Ising, TovJ)
    return Ising

def wolff (Ising, TovJ, nstep):
    Chains = []
    for n in range(nstep):
        site_index = step_1(Ising)
        Ising = step_2(site_index, Ising)
        Ising = step_3(site_index, Ising, TovJ)
        aux = np.array(Ising)
        Chains.append(aux)
    return Chains
    
def simulation_2D_1T (TovJ, L = L):
    t1 = t.time()
    sim_1T = []
    for i in range(nbin):
        Ising = rnd.choice([1, -1], size = (L,L)) #we inizialize a random config for an Ising lattice
        Ising = equilibrate(Ising, TovJ, int(N_mean/3))
        sim_1T.append(np.array(wolff (Ising, TovJ, N_mean)))
    t2 = t.time()
    print('Chain generated, time taken for T/J='+str(TovJ)+' : %.2f'%(t2 - t1))
    print ('Computing m', end="\r")
    t1 = t.time()
    m, err = m_2D(sim_1T, L=L)
    print ('Computing m2', end="\r")
    m2, err2 = m2_2D(sim_1T, L=L)
    t2 = t.time()
    print ('done, time taken for T/J='+str(TovJ)+' : %.1f'%(t2 - t1))
    return [TovJ, m, err, m2, err2]

def simulation_Gamma_1T (TovJ, L = L):
    t1 = t.time()
    sim_1T = []
    for i in range(nbin):
        Ising = rnd.choice([1, -1], size = (L,L)) #we inizialize a random config for an Ising lattice
        Ising = equilibrate(Ising, TovJ, int(N_mean/3))
        sim_1T.append(np.array(wolff (Ising, TovJ, N_mean)))
    t2 = t.time()
    print('Chain generated, time taken for T/J='+str(TovJ)+' : %.2f'%(t2 - t1))
    t1 = t.time()
    #m, err = m_2D(sim_1T, L=L)
    print ('Computing m2', end="\r")
    m2, err2 = m2_2D(sim_1T, L=L)
    print ('Computing m4', end="\r")
    m4, err4 = m4_2D(sim_1T, L=L)
    Gamma = m4/(m2**2)
    t2 = t.time()
    print ('done, time taken for T/J='+str(TovJ)+' : %.1f'%(t2 - t1))
    return [TovJ, Gamma]

#average magnetization per site
def m_1D (filename, TovJ, nbin = 10):
    ms = []
    for i in range(nbin):
        Chain_of_Chains = np.loadtxt(filename)
        x = np.arange(0, N_mean, n_step)
        sums = []
        for index in range(L):
            sums.append(np.average(Chain_of_Chains[x,index])) #For each spin site we average over the samples
        ms.append(np.average(sums)) #then we average over the sites in the Ising chain
    return np.average(ms), np.std(ms) #then we average over the the nbin 'experiments' and extract the std dev

def m_2D (sim, L=L):
    ms = np.average(sim, axis=(1,2,3))
    return np.average(ms), np.std(ms) #then we average over the the nbin 'experiments' and extract the std dev

def m2_2D (sim, L=L):
    m2s_s = np.average(sim, axis=(2,3))**2
    m2s = np.average(m2s_s, axis=1)
    return np.average(m2s), np.std(m2s) #then we average over the the nbin 'experiments' and extract the std dev

def m4_2D (sim, L=L):
    m4s_s = np.average(sim, axis=(2,3))**4
    m4s = np.average(m4s_s, axis=1)
    return np.average(m4s), np.std(m4s)

#Correlation function
def C (r, TovJ, N_mean = 10000):
    Cs = []
    t1 = t.time()
    for i in range(nbin):
        Ising = rnd.choice([1, -1], size = L)
        Ising = equilibrate(Ising, TovJ, int(N_mean/3))
        Chain_of_Chains = np.array(wolff(Ising, TovJ, N_mean))
        x = np.arange(0, N_mean, n_step)
        Cs.append(np.average(Chain_of_Chains[x, 0] * Chain_of_Chains[x, r]) 
                  - np.average(Chain_of_Chains[x, 0])*np.average(Chain_of_Chains[x, r])) # we average over the samples
    t2 = t.time()
    print('time taken for 1 radius: '+str(t2-t1)+' seconds.')
    return np.average(Cs), np.std(Cs) #then we average over the the nbin 'experiments' and extract the std dev
