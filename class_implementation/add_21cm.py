# A script to add the 21cm absorption

# Importing modules
import numpy as np
import matplotlib.pyplot as plt

# Foreground arrays (no 21cm feature)
a = np.loadtxt("0_4W_cluster.txt", delimiter=",")
a1 = np.loadtxt("5_4W_cluster.txt", delimiter=",")
a2 = np.loadtxt("11_4W_cluster.txt", delimiter=",")
a3 = np.loadtxt("17_4W_cluster.txt", delimiter=",")

freq = a[0]

# Defining 21cm feature
amp = 0.5
x0 = 78
width = 18.1
t21 = -amp*np.exp((-4*np.log(2)*(freq-x0)**2)/(width**2))

# Flattened 21cm Gaussian
#amp = 0.5
#x0 = 78
#width = 18
#tau = 7.5

#B = (4.0 * ((freq - x0)**2.0)/width**2) * np.log(-np.log((1.0 + np.exp(-tau))/2.0)/tau)
#t21 =  -amp * (1.0 - np.exp(-tau * np.exp(B)))/(1.0 - np.exp(-tau))

b = np.zeros_like(a)
b1 = np.zeros_like(a1)
b2 = np.zeros_like(a2)
b3 = np.zeros_like(a3)

b[0] = a[0]
b1[0] = a1[0]
b2[0] = a2[0]
b3[0] = a3[0]

# Adding feature to foregrounds
b[1] = a[1] + t21
b1[1] = a1[1] + t21
b2[1] = a2[1] + t21
b3[1] = a3[1] + t21

# Saving to .txt
np.savetxt("0_full_4w_cluster.txt",b,delimiter=",")
np.savetxt("5_full_4w_cluster.txt",b1,delimiter=",")
np.savetxt("11_full_4w_cluster.txt",b2,delimiter=",")
np.savetxt("17_full_4w_cluster.txt",b3,delimiter=",")

# Plotting to check
plt.figure()
plt.subplot(1,3,1)
plt.plot(freq,b[1],"r-",label="0-6hr foreground + 21cm feature")
plt.plot(freq,b1[1],"b-",label="6-12hr foreground + 21cm feature")
plt.plot(freq,b2[1],"g-",label="12-18hr foreground + 21cm feature")
plt.plot(freq,b3[1],"m-",label="18-24hr foreground + 21cm feature")
plt.legend()
plt.subplot(1,3,2)
plt.plot(freq,t21,"r-")
plt.subplot(1,3,3)
plt.plot(freq, b[1] - a[1], "m-",label="0-6hr 21cm feature")
plt.plot(freq, b1[1] - a1[1], "g-",label="6-12hr 21cm feature")
plt.plot(freq, b2[1] - a2[1], "b-",label="12-18hr 21cm feature")
plt.plot(freq, b3[1] - a3[1], "r-",label="18-24hr 21cm feature")
plt.legend()
plt.savefig("cluster_graphs_4w.png")

