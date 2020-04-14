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
data_1 = np.loadtxt("0_full_2w_cluster.txt", delimiter=",")
data_2 = np.loadtxt("11_full_2w_cluster.txt", delimiter=",")
freq = data_1[0]
signal_1 = data_1[1]
signal_2 = data_2[1]
signal = [signal_1, signal_2]
#noise = 0.5e-2
noise_1 = (signal_1 + 200)/(3.879e5)
noise_2 = (signal_2 + 200)/(3.879e5)

# DEFINING MODEL
my_model = md.multi_fg(freq) # model selected from model_database.py

# DEFINING LOG LIKELIHOOD AND PRIORS
def log_likelihood(cube): # log likelihood function
    a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, amp, x0, width = cube
    foregrounds = my_model.foregrounds(cube)
    fg_1 = foregrounds[0]
    fg_2 = foregrounds[1]
    absorption = my_model.signal(cube) 
    model_1 = fg_1 + absorption
    model_2 = fg_2 + absorption
    normalise_1 = 1/(np.sqrt(2*pi*noise_1**2))
    normalise_2 = 1/(np.sqrt(2*pi*noise_2**2)) 
    denominator_1 = 2*noise_1**2
    denominator_2 = 2*noise_2**2
    numerator_1 = (signal[0] - model_1)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    numerator_2 = (signal[1] - model_2)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    loglike_1 = np.sum(np.log(normalise_1) - (numerator_1/denominator_1))
    loglike_2 = np.sum(np.log(normalise_2) - (numerator_2/denominator_2))
    return loglike_1 + loglike_2

def prior(cube): # priors for model parameters
   for i in range(10):
      cube[i]=-2000+2*2000*(cube[i])
   cube[10]=2*cube[10]
   cube[11]=60 + 30*cube[11]
   cube[12]=30*cube[12]
   return cube

multinest_object = multi.multinest_object(data=signal, model=my_model, priors=prior, loglike=log_likelihood)

if __name__ == "__main__":
    multinest_object.solve_multinest()
