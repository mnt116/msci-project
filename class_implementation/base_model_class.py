# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:46:13 2019

@author: matth
"""

# Creating a class structure for different foreground and signal models

# Importing modules
import numpy as np
import matplotlib.pyplot as plt

# Base Model class
class model:
    """
    Base class for a foreground and signal model
    """
    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "Base"
        self.name_sig = "Base"
        self.labels = []  #list of names of parameters
        pass

    def __repr__(self):
        return "(FG:{self.name_fg} + SIG:{self.name_sig} nu=[{np.min(self.freq)}, {np.max(self.freq)}])"

    def __str__(self):
        return "(FG:{self.name_fg} + SIG:{self.name_sig} nu=[{np.min(self.freq)}, {np.max(self.freq)}])"

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

        sky = sig + fg
 
        return self.process(sky, theta)       

    def plot_observation(self, theta, withFG=True, withSIG=True, style='b-'):
        """
        Plots signal over frequency range
        """
        y_vals = self.observation(theta, withFG, withSIG)
        x_vals = self.freq
        style = 'b-'

        plt.figure()
        plt.plot(x_vals, y_vals, style)
        plt.xlabel("Frequency/MHz")
        plt.ylabel("Brightness Temperature/K")
        plt.show()

    def process(self, sky, theta):
        """
        Used to implement ionospheric/instrumental effects
        """
        return sky

    def foreground(self, theta):
        """
        Calculate foreground model
        """
        pass

    def signal(self, theta):
        """
        Calculate 21 cm signal model
        """
        pass
