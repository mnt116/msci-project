# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:32:03 2019

@author: matth
"""

# The runfile for the 21cm mcmc fitting program

# importing modules
import backend as be
import numpy as np

# creating simulated foreground data using EDGES foreground; see edgesfit.py
a0 = 47052.32658584365
a1 = -1844.967333964787
a2 = 29.265792826929875
a3 = -0.21536464292060956
a4 = 0.0006102144740830822
sim_coeffs = np.array([a0,a1,a2,a3,a4]) # simulated foreground coeffs

# creating simulated 21cm data
sim_amp = 0.5 # amplitude
sim_maxfreq = 78 # centre i.e. peak frequency
sim_sigma_hi = 8.1 # width

int_time = 1600 # antennna integration time   

freqfull = np.linspace(51.965332, 97.668457, 50) # frequency linspace 
    
sim_data = be.signal(freqfull, sim_coeffs, sim_maxfreq, sim_amp, sim_sigma_hi, int_time)

pos = np.array([a0,a1,a2,a3,a4,78,0.5,8.6]) # initial values of model parameters
prior_list = [[-1e5,1e5],
              [-4e4,4e4],
              [-500,500],
              [-100,100],
              [-100,100],
              [-1000,1000],
              [0,1000],
              [0,1000]]
n_steps = 500 # number of steps for mcmc
n_walkers = 500 # number of walkers for ensemble sampler

function = be.log_probability # function used to compute log probability

mcmc = be.run_mcmc(pos, n_steps, n_walkers, function, freqfull, sim_data, prior_list) # mcmc chain object
chain = mcmc.get_chain() # unflattened chain
flatchain = mcmc.get_chain(discard=120, thin=15, flat=True) # flattened, thinned chain with burn-in discarded

labels = ["a0","a1","a2","a3","a4","freq","amp","sigma"] # labels for plot

be.plotsimdata(freqfull, sim_data, save=False)
be.plotburnin(chain, labels)
be.plotcorner(flatchain, labels)
be.plotmodels(freqfull, sim_data, flatchain, 100)
be.plotsigmodels(freqfull, sim_data, flatchain, 100)