import numpy as np
import math
import matplotlib.pyplot as plt
import

def readData(dir):
    dir = "Assignment 3/sv.dat"
    data = np.loadtxt(dir, dtype=np.float32, skiprows=1)
    return data

def dataStats(data, label):
    #plt.figure()
    plt.plot(data, color="blue",label=label, linewidth=0.5)
    plt.legend(loc='upper right')
    plt.title('Pound/Dollar daily exchange rates')
    print('The mean  of the ' +label + ' is: ' + str(np.mean(data)))
    print('The variance of the: ' +label + ' is: ' + str(np.var(data)))
    plt.show()

# Read the data
sv_dat = readData("Assignment 3/sv.dat")

# a)
#Data descriptions: plot/mean/var/etacross
dataStats(sv_dat, 'returns')

# b)
xt = np.log(sv_dat**2)
dataStats(xt, 'Log squared returns')

# c)
