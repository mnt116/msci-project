#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
__doc__ = """
Script that does default visualizations (marginal plots, 1-d and 2-d).

Author: Johannes Buchner (C) 2013-2019
"""
import numpy
from numpy import exp, log
import matplotlib.pyplot as plt
import sys, os
import json
import pymultinest
import corner
import model_database as md
from importlib import import_module
import shutil
if len(sys.argv) != 2:
	sys.stderr.write("""SYNOPSIS: %s <output-root> 

	output-root: 	Where the output of a MultiNest run has been written to. 
	            	Example: chains/1-
%s""" % (sys.argv[0], __doc__))
	sys.exit(1)

prefix = sys.argv[1]
print('model "%s"' % prefix)
if not os.path.exists(prefix + "/" + prefix + 'params.json'):
	sys.stderr.write("""Expected the file %sparams.json with the parameter names.
For example, for a three-dimensional problem:

["Redshift $z$", "my parameter 2", "A"]
%s""" % (sys.argv[1], __doc__))
	sys.exit(2)

model_runfile=import_module(prefix[:-1])

parameters = json.load(open(prefix + "/" + prefix + 'params.json'))
n_params = len(parameters)

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = prefix + "/" + prefix)
s = a.get_stats()

json.dump(s, open(prefix + "/" + prefix + 'stats.json', 'w'), indent=4)

print('  marginal likelihood:')
print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
print('  parameters:')
paramlist = [] # for plotting model vs data
for p, m in zip(parameters, s['marginals']):
	lo, hi = m['1sigma']
	med = m['median']
	paramlist.append(med)
	sigma = (hi - lo) / 2
	if sigma == 0:
		i = 3
	else:
		i = max(0, int(-numpy.floor(numpy.log10(sigma))) + 1)
	fmt = '%%.%df' % i
	fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
	print(fmts % (p, med, sigma))

print('creating marginal plot ...')
data = a.get_data()[:,2:]
weights = a.get_data()[:,0]

#mask = weights.cumsum() > 1e-5
mask = weights > 1e-4

corner.corner(data[mask,:], weights=weights[mask], 
	labels=parameters, show_titles=True)

if not os.path.exists(prefix[:-1] + "_results"):
    os.mkdir(prefix[:-1] + "_results")
else:
    for i in os.listdir('.'):
        shutil.rmtree(prefix[:-1] + "_results")
        os.mkdir(prefix[:-1] + "_results")    
plt.savefig(prefix[:-1] + "_results/" + prefix + 'corner.pdf')
plt.savefig(prefix[:-1] + "_results/" + prefix + 'corner.png')
plt.close()


# IMPORTING DATA
freq = model_runfile.freq
obs_signal = model_runfile.signal

# GETTING CONVERGED MODEL
mymodel = model_runfile.my_model
final_vals = numpy.array(paramlist)
mod_signal = mymodel.observation(final_vals)

# CALCULATING RESIDUALS
residuals = (obs_signal-mod_signal)

# CREATING 'ZOOMED IN' PLOT
mod_dip = mymodel.observation(final_vals, withFG=False)
obs_dip = obs_signal - mymodel.observation(final_vals, withSIG=False)

# PLOTTING OBSERVED DATA VS. CONVERGED MODEL
plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt.plot(freq, obs_signal, 'ro', label="observed")
plt.plot(freq, mod_signal, 'b-', label="model")
plt.legend()
plt.title("Model vs. Observed (full range)")
plt.xlabel("Frequency/MHz")
plt.ylabel("Brightness Temperature/K")
plt.subplot(1,3,2)
plt.plot(freq, residuals, 'b-')
plt.title("Residuals (full range)")
plt.xlabel("Frequency/MHz")
plt.subplot(1,3,3)
plt.plot(freq, obs_dip, 'ro', label="residuals foreground only fit")
plt.plot(freq, mod_dip, 'b-', label="full model")
plt.legend()
plt.title("Model vs. Observed (absorption feature)")
plt.xlabel("Frequency/MHz")
plt.ylabel("Brightness Temperature/K")
plt.subplots_adjust(wspace=0.3)
plt.savefig(prefix[:-1] + "_results/" + prefix+"model_vs_observed.png", dpi=200)
