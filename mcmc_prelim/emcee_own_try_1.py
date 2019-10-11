# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 03:12:34 2019

@author: matth
"""

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import emcee, corner
from scipy.optimize import minimize 

# 'true' parameters
x_0_tr = 1
x_1_tr = 12
x_2_tr = 5

# simulate some data
np.random.seed(124)
N=50
x = np.sort(10*np.random.rand(N))
yerr = 10*np.random.rand(N)
y = (x_0_tr) + (x_1_tr)*x + (x_2_tr)*x**2 + np.random.normal(0, yerr**2, N)
x0 = np.linspace(0, 10, 500)

# least-squares fit 
fit = np.polyfit(x, y, 2)
x_0 = fit[2]
x_1 = fit[1]
x_2 = fit[0]

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt='.k', label="data")
plt.plot(x0, (x_0_tr)+(x_1_tr*x0)+(x_2_tr*x0**2), "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, x_0 + x_1*x0 + x_2*x0**2, "b-", label="fit")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

# optimise log likelihood
def log_likelihood(theta, x, y, yerr):
    x_0, x_1, x_2 = theta
    model = x_0 + x_1*x + x_2*x**2
    sigma2 = yerr**2
    return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([x_0_tr, x_1_tr, x_2_tr]) + 0.1*np.random.randn(3)
soln = minimize(nll, initial, args=(x, y, yerr))
x_0_ml, x_1_ml, x_2_ml = soln.x

print("Maximum likelihood estimates:")
print("x_0 = {0:.3f}".format(x_0_ml))
print("x_1 = {0:.3f}".format(x_1_ml))
print("x_2 = {0:.3f}".format(x_2_ml))

plt.figure()
plt.errorbar(x, y, yerr=yerr, fmt='.k', label="data")
plt.plot(x0, x_0_tr+x_1_tr*x0+x_2_tr*x0**2, "k", alpha=0.3, lw=3, label="truth")
plt.plot(x0, x_0 + x_1*x0 + x_2*x0**2, "b-", label="LS fit")
plt.plot(x0, x_0_ml + x_1_ml*x0 + x_2_ml*x0**2, "r--", label="ML fit")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

# MCMC technique
def log_prior(theta):
    x_0, x_1, x_2 = theta
    if -1000.0 < x_0 < 1000.0 and -1000.0 < x_1 < 1000.0 and -1000.0 < x_2 < 1000.0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

pos = 1e-4*np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, 10*yerr))
sampler.run_mcmc(pos, 5000, progress=True)

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["x_0", "x_1", "x_2"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)

# Model vs true for 100 random samples from chain
inds = np.random.randint(len(flat_samples), size=100)
plt.figure()
for ind in inds:
    sample = flat_samples[ind]
    plt.plot(x0, x_0 + x_1*x0 + x_2*x0**2, "g", alpha=0.1)
plt.errorbar(x, y, yerr=yerr, fmt='.k', label="data")
plt.plot(x0, x_0_tr+x_1_tr*x0+x_2_tr*x0**2, "k", alpha=0.3, lw=3, label="truth")
plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y")

fig=corner.corner(flat_samples, labels=labels, truths=[x_0_tr, x_1_tr, x_2_tr])
