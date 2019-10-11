# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 00:44:28 2019

@author: matth
"""

import numpy as np
import emcee, corner

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5*np.dot(diff, np.linalg.solve(cov,diff))

ndim = 5

np.random.seed(42)
means = np.random.rand(ndim)

cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov,cov)

nwalkers = 32
p0 = np.random.rand(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

sampler.run_mcmc(p0, 10000, progress=True)

import matplotlib.pyplot as plt

samples = sampler.get_chain(discard=100, thin=15, flat=True)

fig = corner.corner(samples, labels=['d0','d1','d2','d3','d4'])