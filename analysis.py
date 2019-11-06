import numpy as np
import numpy.random as npr
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
import emcee

fig1file = "figure1_plotdata.csv"

#############################################################
# Test ability to read data from Bowman (2018) and get sensible figure
#############################################################

def figure1():

    data = ascii.read(fig1file)

    #read in data
    dstart = 3
    dend = -2
    freq = data['Frequency [MHz]'][dstart:dend]
    weight = data['Weight'][dstart:dend]
    tsky = data['a: Tsky [K]'][dstart:dend]
    tres1 = data['b: Tres1 [K]'][dstart:dend]
    tres2 = data['c: Tres2 [K]'][dstart:dend]
    tmodel = data[ 'd: Tmodel [K]'][dstart:dend]
    t21 = data['e: T21 [K]'][dstart:dend]

    fig = plt.figure()

    #key sub plots from Figure 1 of Bowman 2018
    ax = fig.add_subplot(2,2,1)
    ax.plot(freq, tsky)
    ax.set_title("Tsky")

    ax = fig.add_subplot(2,2,2)
    ax.plot(freq, tres1)
    ax.set_title("foreground removed residuals")
    
    ax = fig.add_subplot(2,2,3)
    ax.plot(freq, tres2)
    #ax.set_title("residuals")
    
    ax = fig.add_subplot(2,2,4)
    ax.plot(freq, t21)
    #ax.set_title("21cm signal")
    
    plt.show()
    plt.close()

#############################################################
# Signal models
#############################################################

def gaussian(nu, sigma = 10.0, A = 500.0, nu0 = 75.0):
    """
    Gaussian absorption feature
    Here A is interpreted as depth of deepest point
    """
    x = (nu - nu0) / sigma
    t21 = -A * np.exp( - x * x / 2.0)
    return t21
    

def flattenedGaussian(nu, tau = 6.0, w = 20.0, A = 500.0, nu0 = 75.0,):
    """
    Flattened Gaussian model from Bowman (2018)
    """

    B = 4.0 * np.power(nu - nu0, 2.0) / w ** 2
    B *= np.log(-np.log((1.0 + np.exp(-tau))/2.0) / tau)

    t21 = -A * (1.0 - np.exp(-tau * np.exp(B))) / (1.0 - np.exp(-tau))

    #print A, nu0, w, tau

    return t21

def testSignal():
    """
    Simple test plot
    """
    nu = np.linspace(50.0, 100.0)
    nu0 = 75.0

    tau = 6.0
    w = 20.75
    A = 500.0

    t21 = flattenedGaussian(nu, tau, w, A, nu0)
    plt.plot(nu, t21)


    t21 = gaussian(nu, sigma = 8.0, A = A, nu0 = nu0)
    print(t21)
    plt.plot(nu, t21, "--")
    
    plt.show()
    
#############################################################
# Forground models
#############################################################

def foregroundModel(freq, av):
    """
    approximate foreground model from Eq (1) of Bowman 2018
    """

    #central frequency
    #freqc = ( np.max(freq) - np.min(freq) ) / 2.0 + np.min(freq)
    freqc = 75.0
    
    x = freq / freqc

    #Build polynomial model

    tf = av[0] * np.power(x, -2.5)
    tf += av[1] * np.power(x, -2.5) * np.log(x)
    tf += av[2] * np.power(x, -2.5) * np.power(np.log(x), 2.0)
    tf += av[3] * np.power(x, -4.5)
    tf += av[4] * np.power(x, -2.0)
    
    return tf

def foregroundPolynomial(freq, av):
    """
    polynomial foreground model from Eq (2) of Bowman 2018
    """

    #central frequency
    #freqc = ( np.max(freq) - np.min(freq) ) / 2.0 + np.min(freq)
    freqc = 75.0
    
    x = freq / freqc

    #Build polynomial model

    tf =  av[0] * np.power(x, -2.5)
    tf += av[1] * np.power(x, -1.5)
    tf += av[2] * np.power(x, -0.5)
    tf += av[3] * np.power(x,  0.5)
    tf += av[4] * np.power(x,  1.5)
    
    return tf

def testModel():
    """
    Quick plot to check model looks okay
    """
    #read in data
    data = ascii.read(fig1file)
    dstart = 3
    dend = -2
    freq = data['Frequency [MHz]'][dstart:dend]
    tsky = data['a: Tsky [K]'][dstart:dend]

    a = np.zeros(5)
    a[0] = 1560.0
    a[1] = 700.0
    a[2] = -1200.0
    a[3] = 750.0
    a[4] = -175.0

    tf = foregroundPolynomial(freq, a)
    
    fig = plt.figure()

    ax = fig.add_subplot(2,1,1)
    ax.plot(freq, tsky)
    ax.plot(freq, tf, "--")
    ax.set_title("Tsky")

    ax = fig.add_subplot(2,1,2)
    ax.plot(freq, tsky - tf)
    ax.set_title("Tsky - model")
    
    plt.show()
    plt.close()


def fullSignal(freq, av):
    """
    polynomial foreground model from Eq (2) of Bowman 2018

    +

    flattened Gaussian model
    """

    tf = foregroundPolynomial(freq, av[0:5])
    tsig = flattenedGaussian(freq, A = av[5], nu0 = av[6], w = av[7], tau = av[8])

    return tf + tsig

    
def fitting(withsignal = False, usemock = False):
    """
    Fit polynomial to foregrounds
    """

    labels = ["a0", "a1", "a2", "a3", "a4", "A", "nu0", "w", "tau"]

    #read in data
    data = ascii.read(fig1file)
    dstart = 3
    dend = -2
    freq = data['Frequency [MHz]'][dstart:dend]
    weight = data['Weight'][dstart:dend]
    tsky = data['a: Tsky [K]'][dstart:dend]
    tres1 = data['b: Tres1 [K]'][dstart:dend]
    tres2 = data['c: Tres2 [K]'][dstart:dend]
    tmodel = data[ 'd: Tmodel [K]'][dstart:dend]
    t21 = data['e: T21 [K]'][dstart:dend]

    #Noise on measurements in Kelvin
    sigT = 0.01  

    #test with mock data set
    if usemock:
        if withsignal:
            a = [1560.0, 700.0, -1200.0, 750.0, -175.0, 0.5, 75.0, 20.6, 6.0]
            t21 = fullSignal(freq, a)
        else:
            a = [1560.0, 700.0, -1200.0, 750.0, -175.0]
            t21 = foregroundPolynomial(freq, a)

        #Add mock noise - should really be larger at low freq as Tsky^2
        t21 += npr.normal(0.0, sigT, len(t21))

    #Set number of parameters in model
    ndim = 5

    if withsignal:
        ndim += 4

    #log prior
    def lnprior(a):
        """
        uniform prior
        """

        #uniform limits
        limits = [
            [1300.0, 1800.0],
            [450.0, 1000.0],
            [-1500.0, -800.0],
            [350.0, 900.0],
            [-300.0, -50.0],
            [0.0, 1.0],
            [55.0, 90.0],
            [10.0, 35.0],
            [1.0, 20.0]]

        for i in range(len(a)):
            if (a[i] < limits[i][0]) or (a[i] > limits[i][1]):
                return -np.inf

        if withsignal:
            #restrict tau to positive values
            if a[8] < 0:
                return -np.inf

        if 1 == True:
            return 0.0
        
        return -np.inf

    #log likelihood
    def lnlike(a, freq, tsky, ivar):
        """
        Chi squared likelihood with inverse variance ivar
        """

        if withsignal:
            x = tsky - fullSignal(freq, a)
        else:
            x = tsky - foregroundPolynomial(freq, a)

        return -0.5 * np.sum(ivar * x ** 2)

    def lnprob(a, freq, tsky, ivar):
        lp = lnprior(a)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(a, freq, tsky, ivar)

    #guess start - should really use maximum likelihood guess
    a = np.zeros(ndim)
    a[0] = 1560.0
    a[1] = 700.0
    a[2] = -1200.0
    a[3] = 750.0
    a[4] = -175.0

    if withsignal:
        a[5] = 0.5   #A
        a[6] = 75.0    #nu0
        a[7] = 20.6    #w
        a[8] = 6.0     #tau


    #errors on data points (fake errors)
    sigT = 0.01
    ivar = np.ones(len(freq)) / (sigT * sigT)

    #maximum likelihood to find starting position
    print("Finding max likelihood")
    import scipy.optimize as op
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, a, args=(freq, tsky, ivar))
    a_ml = result["x"]

    print("Max likelihood guess= ", a_ml)

    #emcee fitting
    nwalkers = 500

    print("starting guess = ", a)

    p0 = [a + 1.0e-2 * np.random.rand(ndim) for i in range(nwalkers)]

    #run emcee sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[freq, tsky, ivar])
    sampler.run_mcmc(p0, 200, progress=True)



    #get samples - exclude burn in phase and reshape
    burnin = 50
    samples = sampler.get_chain(discard=burnin, thin=15, flat=True)

    #save samples - not a good idea for long chains!
##    np.savetxt("samples.dat", samples)

##    #plot raw samples
##    fig = plt.figure()
##    for i in range(ndim):
##        ax = fig.add_subplot(ndim, 1, i+1)
##        ax.plot(sampler.chain[:, :, i])
##    plt.savefig("samples.eps")
##    plt.close()

    #confidence intervals
    a_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                              axis=0)))

    for i, limit in enumerate(a_mcmc):
        print(labels[i], "=", limit)

    #make a nice corner plot
    import corner
    fig = corner.corner(samples, labels = labels[0:ndim], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    fig.savefig("triangle.png")
    plt.close()



    #project back onto data space
    for asamp in samples[npr.randint(len(samples), size=20)]:
        plt.plot(freq, foregroundPolynomial(freq, asamp)-tsky, color="k", alpha=0.1)
        #plt.plot(freq, tsky, color="r", lw=2, alpha=0.8)

    plt.savefig("foreground_residuals.eps")
    plt.close()
    



