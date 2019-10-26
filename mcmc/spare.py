# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:28:13 2019

@author: matth
"""


def polynomial(freq, coeffs): # computes polynomial function for values in xarr
    coeffs = coeffs
    l = len(coeffs)
    p = np.arange(0,l,1)
    freq_arr = np.transpose(np.multiply.outer(np.full(l,1), freq))
    pwrs = np.power(freq_arr, p)
    ctp = coeffs*pwrs
    return np.sum(ctp,(1)) 

def gaussian(freq, maxfreq, amp, sigma): # defines gaussian absorption feature
    return amp*np.exp((-(freq-maxfreq)**2)/(2*sigma**2))

def thermal_noise(temp, delta_freq, int_time): # defines thermal noise for a given temperature array, specifying width of frequency bin and integration time
    return temp/(np.sqrt(delta_freq*int_time))