"""
To handle different models for 21cm signal and foregrounds it may be useful
to adopt a class structure
"""

import numpy as np

#Default frequency to use
nuFid = np.linspace(50.0, 100.0)

####################################################
# Make mock data
# - this doesn't really belong here, but ok for now
####################################################
def mock(model, theta, withNoise = True):
    """
    Mock observation
    """
    import numpy.random as npr

    #Default signal
    nu = model.nu
    sigFid = model.observation(theta)

    #Default errors
    errFid = thermalNoise(model.foreground(theta))

    #Add noise to signal
    if withNoise:
        sigFid += npr.normal(0.0, errFid) 
    
    return nu, sigFid, errFid


def thermalNoise(Tsky):
    """
    Thermal noise
    """
    # Thermal noise in K

    #noise = 0.005 * np.ones(len(nu))

    #nu0 = 150.0
    #noise = 0.005 * np.power(nu / nu0, -2.7)
    
    epsilon = 1.0e-4   #should really come from bandwith and tobs
    noise = epsilon * Tsky

    return noise

####################################################
# Base class definition
####################################################

class Model:
    """
    Base class - allow for different models, but all with the same
    interface.

    theta = np.array([]) with the parameters
    """

    def __init__(self, nu = nuFid):
        self.nu = nu
        self.name_fg = "Base"
        self.name_sig = "Base"
        self.labels = []  #list of names of parameters
        pass

    def __repr__(self):
        return f"(FG:{self.name_fg} + SIG:{self.name_sig} nu=[{np.min(self.nu)}, {np.max(self.nu)}])"

    def __str__(self):
        return f"(FG:{self.name_fg} + SIG:{self.name_sig} nu=[{np.min(self.nu)}, {np.max(self.nu)}])"

    def observation(self, theta, withFG = True, withSIG = True):
        """
        Return the full modelled observation
        """

        sig = 0.0
        fg =0.0

        if withSIG:
            sig = self.signal(theta)

        if withFG:
            fg = self.foreground(theta)

        return sig + fg

    def foreground(self, theta):
        """ Calculate foreground model
        """
        pass

    def signal(self, theta):
        """
        Calculate 21 cm signal model
        """
        pass

    def priors(self):
        """
        Allow specifying the range of allowed parameter values
        to demark reasonable parameter space
        - Not sure this belongs here
        - should return an [[low1, high1],...] array of values
        for the parameters
        """
        priors = []
        pass


####################################################
# Simple power law with Gaussian absorption model
####################################################

class Simple(Model):
    """
    Simple 21 cm signal class with power law foreground and
    Gaussian absorption trough

    Requires parameters in form
    theta = [t0, nu0, sigma, A, alpha]
    """
    def __init__(self, nu = nuFid):
        self.nu = nu
        self.name_fg = "Power Law"
        self.name_sig = "Gaussian"
        self.labels = ["T0", "nu0", "sigma", "A", "alpha"]
        pass

    def foreground(self, theta):
        """ Calculate foreground model
        """
        nu0 = 150.0
        A = theta[3]
        alpha = theta[4]
        fg = A * np.power(self.nu / nu0, alpha)
        return fg

    def signal(self, theta):
        """
        Calculate 21 cm signal model
        """
        t0 = theta[0]
        nu0 = theta[1]
        sigma = theta[2]
        chi2 = (self.nu - nu0)**2 / sigma / sigma
        t21 = -1.0 * t0 * np.exp(-chi2 / 2.0)
        return t21
    
####################################################
# Model of Bowman (2018)
####################################################

class Bowman(Model):
    """
    Simple 21 cm signal class with
    sum of powerlaw foregrounds and a flattened Gaussian 21cm signal

    Requires parameters in form
    theta = [A, nu0, w, tau, F1, F2, F3, F4, F5]
    """
    def __init__(self, nu = nuFid):
        self.nu = nu
        self.name_fg = "Bowman"
        self.name_sig = "Flat Gaussian"
        self.labels = ["A", "nu0", "w", "tau", "F1", "F2", "F3", "F4", "F5"]
        pass

    def foreground(self, theta):
        """
        Forground model of Bowman (2018) Eq (1)
        """

        freqc = 75.0
    
        x = self.nu / freqc

        #Build polynomial model

        fg =  theta[4] * np.power(x, -2.5)
        fg += theta[5] * np.power(x, -2.5) * np.log(x)
        fg += theta[6] * np.power(x, -2.5) * np.power(np.log(x), 2.0)
        fg += theta[7] * np.power(x, -4.5)
        fg += theta[8] * np.power(x, -2.0)
        
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian from Bowman (2018)
        """
        A = theta[0]
        nu0 = theta[1]
        w = theta[2]
        tau = theta[3]

        B = 4.0 * np.power(self.nu - nu0, 2.0) / w ** 2
        B *= np.log(-np.log((1.0 + np.exp(-tau))/2.0) / tau)
        
        t21 = -A * (1.0 - np.exp(-tau * np.exp(B))) / (1.0 - np.exp(-tau))

        return t21
    
