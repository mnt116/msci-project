# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:26:12 2019

@author: matth
"""

# SIM DATA
plotsimdata = True # plot simulated data
savesimdata = False # save plot of simulated data as simdata.png

# BURN IN
plotburnin = True # plot step-by-step progress of mcmc
saveburnin = False # save plot of step-by-step progress of mcmc as 21cm_burnin.png

# CORNER PLOT
plotcorner = True # plot corner plot of mcmc results
savecorner = False # save corner plot as 21cm_cornerplot.png
 
# MODELS PLOT
plotmodels = True # plot models vs simulated data for 100 random points in chain
savemodels = False # save plot of models vs simulated data as 21cm_modelsplot.png

# SAVE CHAIN
savefullchain = False # choose whether or not to save full mcmc chain to fullchain.txt
saveflatchain = False # choose whether or not to save flattened, thinned chain to flatchain.txt (parameters for discard and thin are set below)

# importing modules
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

# creating simulated foredground data using EDGES foreground; see edgesfit.py
a0 = 47052.32658584365
a1 = -1844.967333964787
a2 = 29.265792826929875
a3 = -0.21536464292060956
a4 = 0.0006102144740830822

freqfull = np.linspace(51.965332, 97.668457, 50) # frequency linspace 

simfore = a0 + a1*freqfull + a2*freqfull**2 + a3*freqfull**3 + a4*freqfull**4 # simulated foreground 

def gaussian(x, x0, A, sigma): # defines gaussian absorption feature
    return (1/(np.sqrt(2*pi*sigma**2)))*A*np.exp((-(x-x0)**2)/(2*sigma**2))

simh = - gaussian(freqfull, 78.0, 10, 8.1) # gaussian absorption dip with parameters comparable to Bowman et al.
simsig = simfore + simh + np.random.normal(0, 0.001, len(freqfull)) # foreground + absorption dip + thermal noise

if plotsimdata == True: # plot of simulated 21cm signal, foreground, and combined 
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(freqfull, simh, 'r.')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Temp [K]")
    plt.title("Simulated 21cm Signal")

    plt.subplot(1,3,2)
    plt.plot(freqfull, simfore, 'r.')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Temp [K]")
    plt.title("Simulated Foreground")

    plt.subplot(1,3,3)
    plt.plot(freqfull, simsig, 'r.')
    plt.xlabel("Freq [MHz]")
    plt.ylabel("Temp [K]")
    plt.title("Full Simulated Signal (to be measured)")

    if savesimdata == True:
        plt.savefig("simdata.png")

# mcmc fitting
tant = simsig # defines antenna temperature measurements, equal to simsig here

def log_likelihood(theta, freq, tant): # log likelihood function for model parameters theta and antenna temperature measurements tant
    p0, p1, p2, p3, p4, amp, maxfreq, sighi = theta # theta takes form of array of model parameters
    coeffs = [p0,p1,p2,p3,p4] # coefficients of model foreground polynomial
    freq_arr = np.transpose(np.multiply.outer(np.full(5,1), freq))
    pwrs = np.power(freq_arr, [0,1,2,3,4])
    ctp = coeffs*pwrs
    T_f = np.sum(ctp,(1)) # model foreground temperature
    T_hi = amp*np.exp(- (freq-maxfreq)**2 / (2*sighi**2)) # model 21cm temperature
    model = T_f + T_hi # total model temperature
    int_time = 16
    sig_therm = tant/(np.sqrt((freq[1]-freq[0])*int_time))
    coeff = 1/(np.sqrt(2*pi*sig_therm**2))
    numerator = (tant - model)**2 # likelihood depends on difference between model and observed temperature in each frequency bin
    denominator = 2*sig_therm**2
    lj = coeff*np.exp(-numerator/denominator) 
    return np.sum(np.log(lj)) # sum over all frequency bins

def log_prior(theta): # defines (uniform) priors for model parameters theta
    p0, p1, p2, p3, p4, amp, maxfreq, sighi = theta # theta takes form of array of model parameters
    if -1e5 < p0 < 1e5 and -4e4 < p1 < 4e4 and -500 < p2 < 500 and -100 < p3 < 100 and -100 < p4 < 100 and -100 < amp < 100 and 0 < maxfreq < 1000 and 1 < sighi < 1000: # uniform priors used for now
        return 0.0 # corresponds to probability of 1
    return -np.inf # corresponds to probability of 0

def log_probability(theta, freq, tant): # combining likelihood and priors
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, freq, tant)


pos = np.array([a0,a1,a2,a3,a4,0.5,78,8.6]) + 1e-4*np.random.randn(32,8) # initial values of model parameters
nwalkers, ndim = pos.shape # number of walkers, and dimensions of sampler

n_steps = 40000 # number of steps for mcmc

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(freqfull, tant))
sampler.run_mcmc(pos, n_steps, progress=True) # runs mcmc; set progress = False for no readout
samples = sampler.get_chain()

if savefullchain == True:
    np.savetxt("fullchain.txt", samples)

if plotburnin == True:
    fig, axes = plt.subplots(8, figsize=(10, 8), sharex=True) # plotting step-by-step progress of mcmc
    labels = ["a0","a1","a2","a3","a4","freq","amp","sigma"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
    if saveburnin == True:
        plt.savefig("21cm_burnin.png")

flat_samples = sampler.get_chain(discard=15000, thin=15, flat=True) # flattened chain; discard burn-in values and thin chain

if saveflatchain == True: # save .txt of flat chain with above parameters
    np.savetxt("flatchain.txt", flat_samples)
  
if plotcorner == True: # corner plot for above mcmc
    fig = corner.corner(flat_samples,labels=labels)
    if savecorner == True:
        plt.savefig("21cm_cornerplot.png")


if plotmodels == True: # plot models vs simulated data for 100 random points in chain
    ff = freqfull
    s_inds = np.random.randint(len(flat_samples), size=100)
    plt.figure() 
    plt.plot(freqfull, simsig, 'k', label = 'truth')
    for i in s_inds:
        sp = flat_samples[i]
        model = sp[0] + sp[1]*ff + sp[2]*ff**2 + sp[3]*ff**3 + sp[4]*ff**4 -gaussian(ff, sp[6], sp[5], sp[7])
        plt.plot(freqfull, model, "g", alpha=0.1)
    plt.legend()
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Temp [K]")
    if savemodels == True:
        plt.savefig("21cm_modelsplot.png")


