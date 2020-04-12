# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:14:51 2019

@author: matth
"""

# runfile for multinest stuff

# importing modules
import model_database as md
import implement_multinest as multi
from math import pi
import numpy as np

# IMPORTING DATA
data = np.loadtxt("edges.txt", delimiter=",")
freq = data[0]
signal = data[1]
t_int = 60*60*107 # 107 hours integration time from Sims et. al 2019
noise = signal/(((data[1]-data[0])*t_int)**0.5)
#noise = signal/3.879*100000 #noise from Sims et. al 2019

# DEFINING MODEL
my_model = md.sims_sine(freq) # model selected from model_database.py

# DEFINING LOG LIKELIHOOD AND PRIORS
def log_likelihood(cube): # log likelihood function
    a0, a1, a2, a3, a4, acal0, acal1, b, P, amp, x0, width, tau = cube
    model = my_model.observation(cube)
    normalise = 1/(np.sqrt(2*pi*noise**2))
    numerator = (signal - model)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    denominator = 2*noise**2
    loglike = np.sum(np.log(normalise) - (numerator/denominator))
    return loglike

def prior(cube): # priors for model parameters
   for i in range(5):
      cube[i]=-5000+2*5000*(cube[i])
   for i in range(5,7):
      cube[i] = -10 + 2*10*cube[i]
   cube[7] = -10 + 2*10*cube[7]
   cube[8] = -50 + 2*50* cube[8]
   cube[9] = 2*cube[9]
   cube[10] = 60 + 30*cube[10]
   cube[11] = 40*cube[11]
   cube[12] = 10*cube[12]
   return cube

multinest_object = multi.multinest_object(data=signal, model=my_model, priors=prior, loglike=log_likelihood)

if __name__ == "__main__":
    multinest_object.solve_multinest()
