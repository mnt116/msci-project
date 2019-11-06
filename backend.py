# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:22:27 2019

@author: matth
"""

# The backend for the 21cm mcmc fitting program

# importing modules
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# defining signal

def absorption(freq, maxfreq, amp, sigma_hi): # signal 21cm absorption dip, defined as a negative gaussian
    return -amp*np.exp((-(freq-maxfreq)**2)/(2*sigma_hi**2))
    
def foreground(freq, coeffs, int_time): # signal foreground
    l = len(coeffs)
    p = np.arange(0,l,1)
    freq_arr = np.transpose(np.multiply.outer(np.full(l,1), freq))
    pwrs = np.power(freq_arr, p)
    ctp = coeffs*pwrs
    return np.sum(ctp,(1)) 

"""
def thermal_noise(freq, temp, int_time): # defines thermal noise for a given temperature array, specifying width of frequency bin and integration time
    return temp/(np.sqrt((freq[1]-freq[0])*(int_time)))
"""
# defining log probability
    
def log_likelihood(theta, freq, simulated, int_time): # log likelihood function for model parameters theta, simulated data, and model data
    a0, a1, a2, a3, a4, maxfreq, amp, sigma_hi = theta # theta takes form of array of model parameters
    coeffs = [a0,a1,a2,a3,a4]
    no_noise = foreground(freq, coeffs, int_time) + absorption(freq, maxfreq, amp, sigma_hi)
    model_noise = thermal_noise(freq, no_noise, int_time)
    model = no_noise + model_noise
    coeff = 1/(np.sqrt(2*pi*model_noise**2))
    numerator = (simulated.full() - model.full())**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    denominator = 2*model_noise**2
    lj = coeff*np.exp(-numerator/denominator) 
    return np.sum(np.log(lj)) # sum over all frequency bins

def log_prior(theta): # defines (uniform) priors for model parameters theta
    a0, a1, a2, a3, a4, amp, maxfreq, sigma_hi = theta # theta takes form of array of model parameters
    if -1e5 < a0 < 1e5 and -4e4 < a1 < 4e4 and -500 < a2 < 500 and -100 < a3 < 100 and -100 < a4 < 100 and -1000 < amp < 1000 and 0 < maxfreq < 1000 and 1 < sigma_hi < 1000: # uniform priors used for now
        return 0.0 # corresponds to probability of 1
    return -np.inf # corresponds to probability of 0

def log_probability(theta, freq, simulated): # combining likelihood and priors
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, freq, simulated)


# mcmc fitting
    
def run_mcmc(pos, n_steps, function, freq, simulated, doprogress = True):
    rand = 0.3*pos*np.random.randn(32,len(pos))
    pos1 = pos+rand
    nwalkers, ndim = pos1.shape # number of walkers, and dimensions of sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, function, args=(freq, simulated))
    sampler.run_mcmc(pos1, n_steps, progress=doprogress) 
    print("Acceptance Fraction:", np.mean(sampler.acceptance_fraction))
    #print("Autocorrelation Time", np.mean(sampler.get_autocorr_time()))
    return sampler

"""
def savefullchain(chain): # this doesn't work yet
    np.savetxt("fullchain.txt", chain)

def saveflatchain(chain, discard, thin):
    get_chain(discard=discardval, thin=thinval, flat=True)
    np.savetxt("flatchain.txt", flatchain)
"""

# plotting graphs

def plotsimdata(freq, sim_signal, save = False): # plot of simulated 21cm signal, foreground, and combined 
    plt.figure()
    
    plt.subplot(2,2,1) # 21cm signal
    plt.plot(freq, sim_signal.absorption(), 'r.')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Temp [K]")
    plt.title("Simulated 21cm Signal")

    plt.subplot(2,2,2) # foreground
    plt.plot(freq, sim_signal.foreground(), 'r.')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Temp [K]")
    plt.title("Simulated Foreground")

    plt.subplot(2,2,3) # thermal noise
    plt.plot(freq, sim_signal.thermal_noise(), 'r.')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Temp [K]")
    plt.title("Simulated Noise")

    plt.subplot(2,2,4) # full simulated signal
    plt.plot(freq, sim_signal.full(), 'r.')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Temp [K]")
    plt.title("Full Simulated Signal (to be measured)")

    if save == True:
        plt.savefig("simdata.png")

def plotcorner(flatchain, labels, save = False): # corner plot for above mcmc
    corner.corner(flatchain,labels=labels)
    if save == True:
        plt.savefig("21cm_cornerplot.png")

def plotmodels(freq, sim_signal, flatchain, size, save=False): # plot models vs simulated data for 100 random points in chain
    s_inds = np.random.randint(len(flatchain), size=size)
    plt.figure() 
    plt.plot(freq, sim_signal.full(), 'k', label = 'truth')
    int_time = sim_signal.getinttime()
    for i in s_inds:
        sp = flatchain[i]
        coeffs = sp[0:-3]
        maxfreq = sp[-3]
        amp = sp[-2]
        sigma = sp[-1]
        plt.plot(freq, signal(freq, coeffs, maxfreq, amp, sigma, int_time).full(), "g", alpha=0.1)
    plt.legend()
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Temp [K]")
    if save == True:
        plt.savefig("21cm_modelsplot.png")

def plotsigmodels(freq, sim_abs, flatchain, size, save=False): # plot models vs simulated data for 100 random points in chain
    s_inds = np.random.randint(len(flatchain), size=size)
    plt.figure() 
    plt.plot(freq, sim_abs, 'k', label = 'truth')
    int_time = sim_signal.getinttime()
    for i in s_inds:
        sp = flatchain[i]
        coeffs = sp[0:-3]
        maxfreq = sp[-3]
        amp = sp[-2]
        sigma = sp[-1]
        plt.plot(freq, signal(freq, coeffs, maxfreq, amp, sigma, int_time).absorption(), "g", alpha=0.1)
    plt.legend()
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Temp [K]")
    plt.ylim(-1,1)
    if save == True:
        plt.savefig("21cm_modelabsorbplot.png")

def plotburnin(chain, labels, save=False):
    fig, axes = plt.subplots(8, figsize=(10, 8), sharex=True) # plotting step-by-step progress of mcmc
    labels = labels
    for i in range(8):
        ax = axes[i]
        ax.plot(chain[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(chain))
        ax.set_xlabel(labels[i])
    if save == True:
        plt.savefig("21cm_burnin.png")
