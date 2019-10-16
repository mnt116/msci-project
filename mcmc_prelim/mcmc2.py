# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:26:12 2019

@author: matth
"""

# importing modules
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
from scipy.optimize import minimize
import corner

# importing EDGES data to simulate background
df = pd.read_csv("edges_data.csv")

freqs = df["freq"]
#weight = df["weight"]
#tmod = df["tmodel"]
tsky = df["tsky"]
#t21 = df["t21"]
#tres1 = df["tres1"]
#tres2 = df["tres2"]
        
# finding form of foreground polynomial for simulated background
N=5
freq1=[]
for i in freqs:
    freq1.append(i)
freq1=np.array(freq1)
freq2=freq1[N:len(tsky)-N]

tsky1=[]
for i in tsky:
    tsky1.append(i)
tsky1=np.array(tsky1)
tsky2=tsky1[N:len(tsky)-N]
        
fit = np.polyfit(freq2, tsky2, 4)

a0 = fit[4]
a1 = fit[3]
a2 = fit[2]
a3 = fit[1]
a4 = fit[0]

freqfull = np.linspace(freq2[0], freq2[-1], 1e4)

skymod = a0 + a1*freqfull + a2*freqfull**2 + a3*freqfull**3 + a4*freqfull**4

plt.figure()
plt.plot(freq2, tsky2, 'ro')
plt.plot(freqfull, skymod, 'b-')
plt.xlabel("Freq [MHz]")
plt.ylabel("Temp [K]")
plt.title("Bowman et al. Sky Data")

# finding form of 21cm signal
def gaussian(x, x0, A, sigma): #defines gaussian absorption feature
    return (1/(np.sqrt(2*pi*sigma**2)))*A*np.exp((-(x-x0)**2)/(2*sigma**2))

hsigmod = - gaussian(freqfull, 78.0, 100, 0.1* 8.1) #gaussian with parameters comparable to Bowman et al.

plt.figure()
plt.plot(freqfull, hsigmod, 'b-')
plt.xlabel("Freq [MHz]")
plt.ylabel("Temp [K]")
plt.title("Simulated 21cm Signal")
plt.savefig("sim21cm.png")

# full simulated signal
simsig = skymod + hsigmod + np.random.normal(0, 0.001, len(freqfull)) #foreground + absorption dip + thermal noise

plt.figure()
plt.plot(freqfull, simsig, 'r.')
plt.xlabel("Freq [MHz]")
plt.ylabel("Temp [K]")
plt.title("Full Simulated Signal (to be measured)")
plt.savefig("fullsimsig.png")

# mcmc fitting
tant = simsig

def log_likelihood(theta, tant):
    p0, p1, p2, p3, p4, amp, maxfreq, sighi = theta
    coeffs = [p0,p1,p2,p3,p4]
    T_f=[]
    for j in range(0, len(freqfull)):
        terms = []
        for i in range(0,5):
            terms.append(coeffs[i]*(freqfull[j])**i)
        T_f.append(np.sum(terms))
    T_hi = amp*np.exp(- (freqfull-maxfreq)**2 / (2*sighi**2))
    model = T_f + T_hi
    #sigma = tant/(np.sqrt(19*60*0.391)) #this was too big/idk why it's here
    coeff = 1/(np.sqrt(2*pi*sighi**2))
    numerator = (tant - model)**2
    denominator = 2*sighi**2
    lj = coeff*np.exp(-numerator/denominator)
    return np.sum(np.log(lj))

def log_prior(theta):
    p0, p1, p2, p3, p4, amp, maxfreq, sighi = theta
    if 4e4 < p0 < 1e5 and -2e3 < p1 < 0 and 0 < p2 < 50 and -0.5 < p3 < 0.5 and 0 < p4 < 1e-3 and 0 < amp < 2 and 50 < maxfreq < 90 and 3 < sighi < 10:
        return 0.0
    return -np.inf

def log_probability(theta, freq, tant):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, tant)


pos = np.array([a0,a1,a2,a3,a4,0.5,78,8.6]) + 1e-4*np.random.randn(32,8)
nwalkers, ndim = pos.shape

n_steps = 2000

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(freqfull, tant))
sampler.run_mcmc(pos, n_steps, progress=True)

fig, axes = plt.subplots(8, figsize=(10, 8), sharex=True)
samples = sampler.get_chain()
labels = ["","","","","","",""]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
plt.savefig("21cm_burnin.png")

"""
nll = lambda *args: -log_likelihood(*args)
initial = np.array([1e-15,1e-15,1e-15,1e-15,1e-15, 2, 7, 1])
soln = minimize(nll, initial, args=(tant))
tmod = soln.x
print(tmod)
"""
flat_samples = sampler.get_chain(discard=1000, thin=3, flat=True)
fig = corner.corner(flat_samples)
plt.savefig("21cm_cornerplot.png")

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
plt.savefig("21cm_mcmcfit.png")
