# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 23:55:54 2019

@author: matth
"""

# importing modules
from __future__ import absolute_import, unicode_literals, print_function
from pymultinest.solve import solve
from pymultinest.analyse import Analyzer
import os
import sys
import shutil
try: os.mkdir('chains')
except OSError: pass


class multinest_object():
    """
    A class to run multinest sampling for a given dataset, model, and priors
    """
    def __init__(self, data, model, priors, loglike, output_prefix = os.path.splitext(os.path.basename(sys.argv[0]))[0]+"-"):
        self.data = data
        self.model = model
        self.priors = priors
        self.loglike = loglike
        self.prefix = output_prefix
    
    def solve_multinest(self, create_analyzer=True):
        """
        Run pymultinest.solve, saving chain and parameter values
        """
        parameters = self.model.labels
        n_params = len(parameters)
        result = solve(LogLikelihood=self.loglike, Prior=self.priors, n_dims=n_params, outputfiles_basename=self.prefix, verbose=True)
        print()
        print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
        print()
        print('parameter values:')
        self.params = []
        for name, col in zip(parameters, result['samples'].transpose()):
            self.params.append(col.mean())
            print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
            
        # make marginal plots by running:
        # $ python multinest_marginals.py chains/3-
        # For that, we need to store the parameter names:
        import json
        if not os.path.exists(self.prefix): 
            os.mkdir(self.prefix)
        else:
            for i in os.listdir('.'):
                shutil.rmtree(self.prefix)
                os.mkdir(self.prefix)
        with open(os.path.join(self.prefix, '%sparams.json' % self.prefix), 'w') as f:
            json.dump(parameters, f, indent=2)
        for i in os.listdir('.'):
            if i.startswith(self.prefix):
                if os.path.isfile(i):
                    shutil.move(i,self.prefix+"/")
        if create_analyzer == True:
            self.analyzer = Analyzer(n_params, outputfiles_basename=self.prefix)
    def get_mode_stats(self):
        return self.analyzer.get_mode_stats()        
