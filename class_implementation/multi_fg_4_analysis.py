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
obs_full_signal = model_runfile.signal

# GETTING CONVERGED MODEL
mymodel = model_runfile.my_model
final_vals = numpy.array(paramlist)

# GETTING MODEL FOREGROUNDS/SIGNAL
mod_foregrounds = mymodel.foregrounds(final_vals)
mod_signal = mymodel.signal(final_vals)

plt.figure()
plt.subplot(1,3,1)
plt.plot(freq,obs_full_signal[0],'ro',label='data_1_observed') 
plt.plot(freq,mod_foregrounds[0]+mod_signal,'b-',label='data_1_fitted')
plt.plot(freq,obs_full_signal[1],'go',label='data_2_observed')
plt.plot(freq,mod_foregrounds[1],'y-',label='data_2_fitted')
plt.plot(freq,obs_full_signal[1],'bo',label='data_3_observed')
plt.plot(freq,mod_foregrounds[1],'y-',label='data_3_fitted')
plt.plot(freq,obs_full_signal[1],'mo',label='data_4_observed')
plt.plot(freq,mod_foregrounds[1],'y-',label='data_4_fitted')
plt.legend()
plt.subplot(1,3,2)
for i in range(4):
    plt.plot(freq,obs_full_signal[i]-mod_foregrounds[i]-mod_signal,'b-')
plt.subplot(1,3,3)
plt.plot(freq,obs_full_signal[0]-mod_foregrounds[0],'ro',label='foregrounds only residual (data 0)')
plt.plot(freq,obs_full_signal[1]-mod_foregrounds[1],'go',label='foregrounds only residual (data 1)')
plt.plot(freq,obs_full_signal[2]-mod_foregrounds[2],'bo',label='foregrounds only residual (data 2)')
plt.plot(freq,obs_full_signal[3]-mod_foregrounds[3],'mo',label='foregrounds only residual (data 3)')
plt.plot(freq,mod_signal,'y-',label='modelled absorption feature')
plt.legend()
plt.savefig("modelvsdata_4w.png")

