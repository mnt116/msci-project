#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import os
try: os.mkdir('chains')
except OSError: pass

import model_class as MC

### EDGES data - need to work out how to feed this in usefully
##import astropy.io.ascii as ascii
##fig1file = "edges_data/figure1_plotdata.csv"
##data = ascii.read(fig1file)
##dstart = 3
##dend = -2
##nu_use = data['Frequency [MHz]'][dstart:dend]
##tsky = data['a: Tsky [K]'][dstart:dend]
### Where do I get EDGES error data?

#Fix frequency range here
#nu_use = np.linspace(50.0, 100.0)
nu_use = MC.nuFid

# select model, fiducial parameters, and prior

case = 2

if case == 1:
	#Power law with a gaussian absorption trough
	model = MC.Simple(nu_use)

	theta = numpy.array([0.4, 70.0, 10.0, 100.0, -2.7])

	priors = numpy.array([[0., 1.],
		  [50.0, 100.0],
		  [1., 40.],
		  [0., 200.],
		  [-3., -2.5]])

elif case == 2:
	#Bowman model with EDGES type paraemters
	model = MC.Bowman(nu_use)

	theta = numpy.array([0.5, 75.0, 20.6, 6.0,
			     1560.0, 700.0, -1200.0, 750.0, -175.0])

	priors = numpy.array([
		[0.0, 1.0],
		[55.0, 90.0],
		[10.0, 35.0],
		[1.0, 20.0],
		[1300.0, 1800.0],
		[450.0, 1000.0],
		[-1500.0, -800.0],
		[350.0, 900.0],
		[-300.0, -50.0]])

else:
	raise Exception("Case not defined")

	
############# Create mock observations #############

#create mock data from fiducial parameters
nuFid, sigFid, errFid = MC.mock(model, theta)

#save data set
data = numpy.array([nuFid, sigFid, errFid])
numpy.savetxt(f"{case}-mydata.dat", data.T)

#save fiducial parameters and prior
data = numpy.array([theta, priors[:,0], priors[:,1]])
numpy.savetxt(f"{case}-myparam.dat", data.T)


############# Below shouldn't require change #############

#multinest requests the transform of a [0,1] unit cube prior
def myprior(p):
	#priors [lower, upper] - assume uniform prior in this range
	
	for i in range(len(p)):
		p[i] = (priors[i][0] + p[i]*(priors[i][1]-priors[i][0]))

	#print(p)
	return p

def myloglike(p):
	#Let's use a simple chi2 for now
	obs = model.observation(p)
	chi2 = numpy.power((obs - sigFid)/errFid, 2.0)
	chi2 = chi2.sum()
	return -chi2/2.0


# number of dimensions our problem has
#parameters = ["T0", "nu0", "sigma", "A", "alpha"]
parameters = model.labels
n_params = len(parameters)
# name of the output files
#prefix = "chains/1-"
prefix = f"chains/{case}-"

# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, 
	n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)

#run the code for pictures - NEEDS TESTING
#os.system("pythonPort ./multinest_marginals.py chains/1-")

