"""
Visualise output from MultiNest

Need to rewrite to mesh with class structure
"""

import model_class as MC
import os
import matplotlib.pyplot as plt
import numpy as np
import pymultinest as pm


def project_chain(case=1, saveFig = False):
    """
    Take points from chain and project them back into signal space
    """

    basedir = f"chains/{case}-"
    if case ==1:
        model = MC.Simple()
    elif case ==2:
        model = MC.Bowman()

    #Find best fitting parameters
    a = pm.Analyzer(len(model.labels), f"chains/{case}-")
    theta = a.get_best_fit()['parameters']
    print(theta)

    #data = np.loadtxt(basedir + "ev.dat")
    #indx = 10002
    #ndim = len(model.labels)
    #theta = data[indx, 0:ndim]
    #ev = data[indx, ndim+2]

    T21 = model.observation(theta)
    sig = model.observation(theta, withFG = False)
    fg = model.observation(theta, withSIG = False)
    nu = model.nu

    #fiducial
    obs = np.loadtxt(f"{case}-myparam.dat")
    thetaFid = obs[:,0]
    T21f = model.observation(thetaFid)
    sigf = model.observation(thetaFid, withFG = False)
    fgf = model.observation(thetaFid, withSIG = False)
    nuf = model.nu

    #Make plot with three panels
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(nu, T21)
    ax1.plot(nuf, T21f,":")

    #Plot 21cm signal
    ax2 = fig.add_subplot(312)
    ax2.plot(nu, sig)
    ax2.plot(nuf, sigf,":")

    #Plot residuals
    ax3 = fig.add_subplot(313)
    obs = np.loadtxt(f"{case}-mydata.dat")
    T21Fid = obs[:,1]
    err = obs[:,2]

    ax3.plot(nu, T21 - T21Fid)
    ax3.plot(nu, sig - sigf,":")
    ax3.errorbar(nu, np.zeros(len(nu)), err)

    plt.xlabel("Freqency [MHz]")

    if saveFig:
        plt.savefig(f"{case}-summary.eps")
    else:
        plt.show()
    
    return


def contours():
    """
    Calculate confidence intevals on model from the MultiNest chain samples

    I'll ignore priors, so this is from liklihood only

    Not sure how most efficiently to do this. Or do Gaussian errors and 1-std
    I can only think of gridding the whole display area and plotting the
    histogram of probability

    Sure someone else has done this already
    """

    data = np.loadtxt(basedir + "ev.dat")

    all = []
    ndim = len(model.labels)
    for i in range(len(data)):
        theta = data[i, 0:ndim]
        like =  data[i, ndim+1]
        ev = data[i, ndim+2]

        prob = like/ev

        T21 = model.observation(theta)

        Tmean += T21 * prob

        #Now do something clever...
        print("???")
