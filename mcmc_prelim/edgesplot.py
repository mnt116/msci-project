# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:17:33 2019

@author: matth
"""

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("edges_data.csv")

freq = df["freq"]
weight = df["weight"]
tmodel = df["tmodel"]
tsky = df["tsky"]
t21 = df["t21"]
tres1 = df["tres1"]
tres2 = df["tres2"]

datalist = [tsky, tres1, tres2, tmodel, t21]
fig, axes = plt.subplots(5, figsize=(7, 8), sharex=True)
for i in range(0,5):
    ax = axes[i]
    ax.plot(freq, datalist[i])
    ax.set_xlim(52, 98)
    ax.set_ylabel("Temp [K]")
    if i == 4:
        ax.set_xlabel("Frequency [MHz]")
plt.savefig("bowman-plots.png")
        
N=5
freq2=freq[N:len(tsky)-N]
tsky2=tsky[N:len(tsky)-N]
        
fit = np.polyfit(freq2, tsky2, 4)

a0 = fit[4]
a1 = fit[3]
a2 = fit[2]
a3 = fit[1]
a4 = fit[0]

skymod = a0 + a1*freq2 + a2*freq2**2 + a3*freq2**3 + a4*freq2**4

plt.figure()
plt.plot(freq2, skymod, 'b-')
plt.plot(freq2, tsky2, 'ro')
plt.xlabel("Freq [MHz]")		
plt.ylabel("Temp [K]")
plt.savefig("sim-foregd.png")
