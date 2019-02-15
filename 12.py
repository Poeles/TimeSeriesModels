import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as st

class Nile:
    def __init__(self):
        self.data = [i.strip().split() for i in open("DK-data/Nile.dat").readlines()]
        self.data.pop(0)

        self.y = np.zeros(len(self.data))

        self.n = len(self.y)

        for i in range(self.n):
            self.y[i] = int(self.data[i][0])

        # initializations Kalman Filter
        self.a1 = 0
        self.P1 = 10 ** 7
        self.vareta = 1469.1
        self.vareps = 15099

        # initializations Kalman Smoother
        self.rn = 0
        self.Nn = 0
        # the label for the x-axis
        self.xYears = ["", 1880, "", 1900, "", 1920, "", 1940, "", 1960]

    def KalmanFilter(self):
        self.v = np.zeros(self.n)
        self.f = np.zeros(self.n)
        self.k = np.zeros(self.n)

        self.a = np.zeros(self.n + 1)
        self.a[0] = self.a1
        self.P = np.zeros(self.n + 1)
        self.P[0] = self.P1

        for t in range(self.n):
            self.v[t] = self.y[t] - self.a[t]
            self.f[t] = self.P[t] + self.vareps
            self.k[t] = self.P[t]/self.f[t]
            self.a[t+1] = self.a[t] + self.k[t]*self.v[t]
            self.P[t+1] = self.k[t]*self.vareps + self.vareta

    def KalmanSmoother(self):
        self.r = np.zeros(self.n + 1)
        self.N = np.zeros(self.n + 1)
        self.alphahat = np.zeros(self.n + 1)
        self.V = np.zeros(self.n + 1)

        self.r[self.n] = self.rn
        self.N[self.n] = self.Nn
        for i in range(self.n + 1):
            t = self.n - i - 1
            self.r[t - 1] = self.v[t]/self.f[t] + (1 - self.k[t]) * self.r[t]
            self.N[t - 1] = 1/self.f[t] + self.N[t] * (1 - self.k[t]) ** 2
            self.alphahat[t] = self.a[t] + self.P[t] * self.r[t-1]
            self.V[t] = self.P[t] - (self.P[t] ** 2) * self.N[t]

        print(self.V)

    def fig1(self):
        """ 2.1.i """
        lowerboundCI = np.zeros(self.n)
        upperboundCI = np.zeros(self.n)
        for i in range(self.n):
            lowerboundCI[i] = self.a[i + 1] - 1.645*np.sqrt(self.P[i + 1])
            upperboundCI[i] = self.a[i + 1] + 1.645*np.sqrt(self.P[i + 1])

        x = np.linspace(0,10,self.n)

        plt.figure()
        plt.plot(x, self.a[1:], color="blue", label=r'$\alpha_t$', linewidth=0.5)
        plt.plot(x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.plot(x, lowerboundCI, color="red", label='90% Confidence Interval', linewidth=0.5, linestyle="dashed")
        plt.plot(x, upperboundCI, color="red", linewidth=0.5, linestyle="dashed")
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state ' + r'$\alpha_t$ ' + 'and its 90% confidence intervals',fontsize=12)
        plt.draw()

        """ 2.1.ii """
        plt.figure()
        plt.plot(x, self.P[1:], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state variance ' + r'$P_t$',fontsize=12)
        plt.draw()

        """ 2.1.iii """
        self.v[0] = 0
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(x, self.v, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Prediction errors ' + r'$v_t$',fontsize=12)
        plt.draw()

        """ 2.1.iv """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.ylim(bottom=20000, top=32500)
        plt.plot(x, self.f, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Prediction errors ' + r'$F_t$',fontsize=12)
        plt.draw()

    def fig2(self):
        """ 2.2.i """
        lowerboundCI = np.zeros(self.n)
        upperboundCI = np.zeros(self.n)
        for i in range(self.n):
            lowerboundCI[i] = self.alphahat[i + 1] - 1.645*np.sqrt(self.V[i + 1])
            upperboundCI[i] = self.alphahat[i + 1] + 1.645*np.sqrt(self.V[i + 1])

        x = np.linspace(0,10,self.n)

        plt.figure()
        plt.plot(x, self.alphahat[1:], color="blue", label='Smoothed state', linewidth=0.5)
        plt.plot(x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.plot(x, lowerboundCI, color="red", label='90% Confidence Interval', linewidth=0.5, linestyle="dashed")
        plt.plot(x, upperboundCI, color="red", linewidth=0.5, linestyle="dashed")
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Nile data and output of state smoothing recursion',fontsize=12)
        plt.draw()

        """ 2.2.ii """
        plt.figure()
        plt.plot(x, self.V[1:], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state variance ' + r'$V_t$',fontsize=12)
        plt.draw()

        """ 2.2.iii """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(x, self.r[:-1], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothing cumulant ' + r'$r_t$',fontsize=12)
        plt.draw()

        """ 2.2.iv """
        plt.figure()
        plt.plot(x, self.N[:-1], color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothing variance cumulant ' + r'$N_t$',fontsize=12)
        plt.draw()



def main():
    nile = Nile()
    nile.KalmanFilter()
    nile.KalmanSmoother()
    #nile.fig1()
    nile.fig2()

    plt.show()

if __name__ == "__main__":
    main()
