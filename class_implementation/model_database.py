# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:37:16 2019

@author: matth
"""

# importing modules
import base_model_class as bmc
import numpy as np

# DATABASE OF DIFFERENT FOREGROUND AND 21CM MODELS ============================
class logpoly_plus_gaussian(bmc.model):
    """
    A log polynomial foreground up to 4th order
    and a gaussian absorption for 21cm signal

    Requires parameters in form
    theta = [a0,a1,a2,a3,a4,amp,x0,width]
    """
    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "log_poly_4"
        self.name_sig = "gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width"]
        pass

    def foreground(self, theta):
        """
        Log polynomial foreground up to 4th order
        """
        freq_0 = 75 # SORT THIS OUT!!! pivot scale
        coeffs = theta[0:-3]
        l = len(coeffs)
        p = np.arange(0,l,1)
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        log_arr = np.log(normfreq)
        pwrs = np.power(log_arr, p)
        ctp = coeffs*pwrs
        log_t = np.sum(ctp,(1))
        fg = np.exp(log_t)
        return fg

    def signal(self, theta): # signal 21cm absorption dip, defined as a negative gaussian
        amp = theta[-3]
        x0 = theta[-2]
        width = theta[-1]
        t21 = -amp*np.exp((-4*np.log(2)*(self.freq-x0)**2)/(width**2))
        return t21

# =============================================================================

class bowman(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.2

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class bowman_physical(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.3

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        normfreq = self.freq/freq_0
        fg = coeffs[0]*np.power(normfreq,-2.5) + coeffs[1]*np.power(normfreq, -2.5)*np.log(normfreq) + coeffs[2]*np.power(normfreq,-2.5)*np.power(np.log(normfreq), 2.0) + coeffs[3]*np.power(normfreq, -4.5) + coeffs[4]*np.power(normfreq, -2.0)
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class hills_physical(bmc.model):
    """
    Model used by Hills in 2019 paper eq.6

    Requires parameters in form
    theta = [b0, b1, b2, b3, Te, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial + ionosphere"
        self.name_sig = "flat gaussian"
        self.labels = ["b0","b1","b2","b3","Te","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        normfreq = self.freq/freq_0
        pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2.0)) + coeffs[4]*(1 - np.exp(-coeffs[3]*np.power(normfreq, -2.0)))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class hills_linear(bmc.model):
    """
    Model used by Hills in 2019 paper eq.6

    Requires parameters in form
    theta = [b0, b1, b2, b3, b4, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial + ionosphere"
        self.name_sig = "flat gaussian"
        self.labels = ["b0","b1","b2","b3","b4","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Linearised physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        normfreq = self.freq/freq_0
        pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2.0)) + coeffs[4]*(np.power(normfreq, -2.0))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# =============================================================================

class hills_sine(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, a5, amp, phi, l]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_5th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","a5","amp","phi","l"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 5th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-3]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Sine function
        """
        amp = theta[-3]
        phi = theta[-2]
        l = theta[-1]

        t21 =  amp * np.sin(2*np.pi*self.freq/l + phi)
        return t21

# ===============================================================================

class freenoise_bowman(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.2, with noise as free parameter

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-5]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-5]
        x0 = theta[-4]
        width = theta[-3]
        tau = theta[-2]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =============================================================================

class freenoise_bowman_physical(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.3

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width, tau, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","tau","noise"]
        pass

    def foreground(self, theta):
        """
        Physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-5]
        normfreq = self.freq/freq_0
        fg = coeffs[0]*np.power(normfreq,-2.5) + coeffs[1]*np.power(normfreq, -2.5)*np.log(normfreq) + coeffs[2]*np.power(normfreq,-2.5)*np.power(np.log(normfreq), 2.0) + coeffs[3]*np.power(normfreq, -4.5) + coeffs[4]*np.power(normfreq, -2.0)
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-5]
        x0 = theta[-4]
        width = theta[-3]
        tau = theta[-2]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =============================================================================

class freenoise_hills_physical(bmc.model):
    """
    Model used by Hills in 2019 paper eq.6

    Requires parameters in form
    theta = [b0, b1, b2, b3, Te, amp, x0, width, tau, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial + ionosphere"
        self.name_sig = "flat gaussian"
        self.labels = ["b0","b1","b2","b3","Te","amp","x0","width","tau","noise"]
        pass

    def foreground(self, theta):
        """
        Physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-5]
        normfreq = self.freq/freq_0
        pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2.0)) + coeffs[4]*(1 - np.exp(-coeffs[3]*np.power(normfreq, -2.0)))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-5]
        x0 = theta[-4]
        width = theta[-3]
        tau = theta[-2]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =============================================================================

class freenoise_hills_linear(bmc.model):
    """
    Model used by Hills in 2019 paper eq.6

    Requires parameters in form
    theta = [b0, b1, b2, b3, b4, amp, x0, width, tau, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "physical polynomial + ionosphere"
        self.name_sig = "flat gaussian"
        self.labels = ["b0","b1","b2","b3","b4","amp","x0","width","tau","noise"]
        pass

    def foreground(self, theta):
        """
        Linearised physical foreground model
        """
        freq_0 = 75.0
        coeffs = theta[0:-5]
        normfreq = self.freq/freq_0
        pwr = -2.5 + coeffs[1] + coeffs[2]*np.log(normfreq)
        fg = coeffs[0]*np.power(normfreq, pwr)*np.exp(-coeffs[3]*np.power(normfreq,-2.0)) + coeffs[4]*(np.power(normfreq, -2.0))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-5]
        x0 = theta[-4]
        width = theta[-3]
        tau = theta[-2]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =============================================================================

class freenoise_hills_sine(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, a5, amp, phi, l, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_5th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","a5","amp","phi","l","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 5th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Sine function
        """
        amp = theta[-4]
        phi = theta[-3]
        l = theta[-2]

        t21 =  amp * np.sin(2*np.pi*self.freq/l + phi)
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise


# =============================================================================

class freenoise_hills_6th(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, a5, amp, phi, l, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_5th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","a5","amp","x0","width","tau","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 5th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-5]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-5]
        x0 = theta[-4]
        width = theta[-3]
        tau = theta[-2]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =============================================================================

class sims_sine(bmc.model):
    """
    Model used by Sims in 2019

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, acal0, acal1, b, P, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_4th + calibration"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","acal0","acal1","b","P","amp","x0","width","tau"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-8]
        acal0 = theta[-8]
        acal1 = theta[-7]
        b = theta[-6]
        P = theta[-5]
        l = len(coeffs)
        pw = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, pw)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        cal = np.power(self.freq/freq_0, b)*(acal0*np.sin(2*np.pi*self.freq/P) + acal1*np.cos(2*np.pi*self.freq/P))
        return fg + cal

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        tau = theta[-1]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

# ================================================================================================================

class multi_fg(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.2

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, amp, x0, width]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial"
        self.name_sig = "gaussian"
        self.labels = ["a0","a1","a2","a3","a4","b0", "b1", "b2", "b3","b4","amp","x0","width"]
        pass

    def foregrounds(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = [theta[0:5], theta[5:10]]
        l = len(coeffs[0])
        p = np.arange(0,l,1)
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        log_arr = np.log(normfreq)
        pwrs = np.power(log_arr, p)
        ctp0 = coeffs[0]*pwrs
        log_fg0 = np.sum(ctp0, (1))
        fg0 = np.exp(log_fg0)
        ctp1 = coeffs[1]*pwrs
        log_fg1 = np.sum(ctp1, (1))
        fg1 = np.exp(log_fg1)
        fg = [fg0, fg1]
        return fg

    def signal(self, theta): # signal 21cm absorption dip, defined as a negative gaussian
        amp = theta[-3]
        x0 = theta[-2]
        width = theta[-1]
        t21 = -amp*np.exp((-4*np.log(2)*(self.freq-x0)**2)/(width**2))
        return t21


# =============================================================================

class multi_fg_4(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.2
    Multiple foregrounds with same signal
    Requires parameters in form
    theta = [a0,a1,a2,a3,a4,b0,b1,b2,b3,b4,c0,c1,c2,c3,c4,d0,d1,d2,d3,d4,amp,x0,width]
    """
    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "log_poly_4"
        self.name_sig = "gaussian"
        self.labels = ["a0","a1","a2","a3","a4","b0", "b1", "b2", "b3","b4","c0","c1","c2","c3","c4","d0","d1","d2","d3","d4","amp","x0","width"]
        pass

    def foregrounds(self, theta):
        """
        Log polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = [theta[0:5],theta[5:10],theta[10:15],theta[15:20]]
        l = len(coeffs[0])
        p = np.arange(0,l,1)
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        log_arr = np.log(normfreq)
        pwrs = np.power(log_arr, p)
        ctp0 = coeffs[0]*pwrs
        log_fg0 = np.sum(ctp0, (1))
        fg0 = np.exp(log_fg0)
        ctp1 = coeffs[1]*pwrs
        log_fg1 = np.sum(ctp1, (1))
        fg1 = np.exp(log_fg1)
        ctp2 = coeffs[2]*pwrs
        log_fg2 = np.sum(ctp2, (1))
        fg2 = np.exp(log_fg2)
        ctp3 = coeffs[3]*pwrs
        log_fg3 = np.sum(ctp3, (1))
        fg3 = np.exp(log_fg3)
        fg = [fg0, fg1, fg2, fg3]
        return fg

    def signal(self, theta): # signal 21cm absorption dip, defined as a negative gaussian
        amp = theta[-3]
        x0 = theta[-2]
        width = theta[-1]
        t21 = -amp*np.exp((-(self.freq-x0)**2)/(2*width**2))
        return t21


# =============================================================================

class multi_fg_1(bmc.model):
    """
    Model used by Bowman in 2018 paper eq.2

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, x0, width]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial"
        self.name_sig = "gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-3]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        log_fg = np.sum(ctp, (1))
        fg = np.exp(log_fg)
        return fg

    def signal(self, theta): # signal 21cm absorption dip, defined as a negative gaussian
        amp = theta[-3]
        x0 = theta[-2]
        width = theta[-1]
        t21 = -amp*np.exp((-(self.freq-x0)**2)/(2*width**2))
        return t21

# =================================================================================================

class freenoise_logpoly_plus_gaussian(bmc.model):
    """
    A log polynomial foreground up to 4th order
    and a gaussian absorption for 21cm signal

    Requires parameters in form
    theta = [a0,a1,a2,a3,a4,amp,x0,width,noise]
    """
    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "log_poly_4"
        self.name_sig = "gaussian"
        self.labels = ["a0","a1","a2","a3","a4","amp","x0","width","noise"]
        pass

    def foreground(self, theta):
        """
        Log polynomial foreground up to 4th order
        """
        freq_0 = 75 
        coeffs = theta[0:-4]
        l = len(coeffs)
        p = np.arange(0,l,1)
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        log_arr = np.log(normfreq)
        pwrs = np.power(log_arr, p)
        ctp = coeffs*pwrs
        log_t = np.sum(ctp,(1))
        fg = np.exp(log_t)
        return fg

    def signal(self, theta): # signal 21cm absorption dip, defined as a negative gaussian
        amp = theta[-4]
        x0 = theta[-3]
        width = theta[-2]
        
        t21 = -amp*np.exp((-4*np.log(2)*(self.freq-x0)**2)/(width**2))

        return t21


    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =================================================================================================

class freenoise_6th_nosig(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, a5, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_5th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","a5","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 5th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-1]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise
# =================================================================================================

class freenoise_4th_nosig(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_3th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 5th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-1]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =================================================================================================

class freenoise_5th_nosig(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_4th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 5th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-1]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =================================================================================================

class freenoise_7th_nosig(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, a5, a6, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_5th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","a5","a6","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 6th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-1]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise

# =============================================================================

class freenoise_sims_sine(bmc.model):
    """
    Model used by Sims in 2019

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, acal0, acal1, b, P, amp, x0, width, tau]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_4th + calibration"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","acal0","acal1","b","P","amp","x0","width","tau","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:5]
        acal0 = theta[5]
        acal1 = theta[6]
        b = theta[7]
        P = theta[8]
        l = len(coeffs)
        pw = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, pw)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        cal = np.power(self.freq/freq_0, b)*(acal0*np.sin(2*np.pi*self.freq/P) + acal1*np.cos(2*np.pi*self.freq/P))
        return fg + cal

    def signal(self, theta):
        """
        Flattened Gaussian
        """
        amp = theta[-5]
        x0 = theta[-4]
        width = theta[-3]
        tau = theta[-2]

        B = (4.0 * ((self.freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)

        t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise


# =============================================================================

class freenoise_sims_sine_nosig(bmc.model):
    """
    Model used by Sims in 2019

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, acal0, acal1, b, P, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_4th + calibration"
        self.name_sig = "flat gaussian"
        self.labels = ["a0","a1","a2","a3","a4","acal0","acal1","b","P","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:5]
        l = len(coeffs)
        pw = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, pw)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Calibration temperature
        """
        freq_0 = 75.0
        acal0 = theta[5]
        acal1 = theta[6]
        b = theta[7]
        P = theta[8]

        t21 = np.power(self.freq/freq_0, b)*(acal0*np.sin(2*np.pi*self.freq/P) + acal1*np.cos(2*np.pi*self.freq/P))
        return t21

    def noise(self, theta):
        """
        Noise as a free parameter
        """
        noise = theta[-1]
        return noise

# ============================================================================================================

class freenoise_hills_sine_5th(bmc.model):
    """
    Model used by Hills in 2018

    Requires parameters in form
    theta = [a0, a1, a2, a3, a4, amp, phi, l, noise]
    """

    def __init__(self, freq):
        self.freq = freq
        self.name_fg = "polynomial_5th"
        self.name_sig = "sine function"
        self.labels = ["a0","a1","a2","a3","a4","amp","phi","l","noise"]
        pass

    def foreground(self, theta):
        """
        Linear polynomial foreground up to 4th order
        """
        freq_0 = 75.0
        coeffs = theta[0:-4]
        l = len(coeffs)
        p = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
        freq_arr = np.transpose(np.multiply.outer(np.full(l,1), self.freq))
        normfreq = freq_arr/freq_0
        pwrs = np.power(normfreq, p)
        ctp = coeffs*pwrs
        fg = np.sum(ctp, (1))
        return fg

    def signal(self, theta):
        """
        Sine function
        """
        amp = theta[-4]
        phi = theta[-3]
        l = theta[-2]

        t21 =  amp * np.sin(2*np.pi*self.freq/l + phi)
        return t21

    def noise(self, theta):
        """
        Noise free parameter
        """
        noise = theta[-1]
        return noise




# add more models here
