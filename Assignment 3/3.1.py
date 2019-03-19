import numpy as np
import math
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======
import
>>>>>>> 33d6fc7fd1371924788a94faf15bf94fcb41f8ce

def readData(dir):
    data = np.loadtxt(dir, dtype=np.float32, skiprows=1)
    return data

def dataStats(data, label):
    #plt.figure()
    plt.plot(data, color="blue",label=label, linewidth=0.5)
    plt.legend(loc='upper right')
    plt.title('Pound/Dollar daily exchange rates')
    print('The mean  of the ' +label + ' is: ' + str(np.mean(data)))
    print('The stdev of the: ' +label + ' is: ' + str(np.std(data)))
    print('The variance of the: ' +label + ' is: ' + str(np.var(data)))
    plt.show()

# Read the data
sv_dat = readData("Assignment 3/sv.dat")

# a)
#Data descriptions: plot/mean/var/etacross
dataStats(sv_dat, 'returns')

# b)
<<<<<<< HEAD
xt = np.log((sv_dat - (np.mean(sv_dat)))**2)
dataStats(xt, 'Log squared demeaned returns')
=======
xt = np.log(sv_dat**2)
dataStats(xt, 'Log squared returns')
>>>>>>> 33d6fc7fd1371924788a94faf15bf94fcb41f8ce

# c)
