"""
Plots figure 2.1 to 2.3
"""

import numpy as np
import math
import matplotlib.pyplot as plt

class Nile:
    def __init__(self):
        self.data = [i.strip().split() for i in open("DK-data/Nile.dat").readlines()]
        self.data.pop(0)

        self.y = np.zeros(len(self.data))

        self.n = len(self.y)

        for i in range(self.n):
            self.y[i] = int(self.data[i][0])

        self.vareta = 1469.1
        self.vareps = 15099

        # initializations Kalman Filter
        self.T = float(input("Given the AR(1) + Noise model. Pick the value for phi in [-1, 1]. If LLM, phi = 1. "))

        self.a1 = 0

        if abs(self.T == 1):
            self.P1 = 10 ** 7
        elif abs(self.T < 1):
            self.P1 = self.vareta/(1 - self.T**2)
        else:
            raise ValueError("The phi you've picked is not within the feasible range!")


        # initializations Kalman Smoother
        self.rn = 0
        self.Nn = 0
        # the label for the x-axis
        self.xYears = ["", 1880, "", 1900, "", 1920, "", 1940, "", 1960]
        self.x = np.linspace(0,10,self.n)

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

        self.a = self.a[1:]
        self.P = self.P[1:]

    def KalmanSmoother(self):
        self.r = np.zeros(self.n)
        self.N = np.zeros(self.n)
        self.alphahat = np.zeros(self.n)
        self.V = np.zeros(self.n)

        self.r[self.n - 1] = self.rn
        self.N[self.n - 1] = self.Nn
        for i in range(self.n - 1):
            t = self.n - 1 - i
            self.r[t - 1] = self.v[t]/self.f[t] + (self.T - self.k[t]) * self.r[t]
            self.N[t - 1] = 1/self.f[t] + self.N[t] * (self.T - self.k[t]) ** 2

        for i in range(self.n):
            t = self.n - 1 - i
            self.alphahat[t] = self.a[t] + self.P[t] * self.r[t-1]
            self.V[t] = self.P[t] - (self.P[t] ** 2) * self.N[t]

    def DisturbanceSmoother(self):
        self.mu = np.zeros(self.n)
        self.epshat = np.zeros(self.n)
        self.varepshatgiveny = np.zeros(self.n)
        self.D = np.zeros(self.n)
        self.etahat = np.zeros(self.n)
        self.varetagiveny = np.zeros(self.n)

        for t in range(self.n):
            self.mu[t] = self.v[t]/self.f[t] - self.k[t] * self.r[t]
            self.epshat[t] = self.vareps * self.mu[t]

            self.D[t] = 1/self.f[t] + (self.k[t]**2) * self.N[t]
            self.varepshatgiveny[t] = self.vareps - (self.vareps**2) * self.D[t]

            self.etahat[t] = self.vareta * self.r[t]
            self.varetagiveny[t] = self.vareta - (self.vareta**2) * self.N[t]

    def fig1(self):
        """ 2.1.i """
        lowerboundCI = np.zeros(self.n)
        upperboundCI = np.zeros(self.n)
        for i in range(self.n):
            lowerboundCI[i] = self.a[i] - 1.645*np.sqrt(self.P[i])
            upperboundCI[i] = self.a[i] + 1.645*np.sqrt(self.P[i])

        plt.figure()
        plt.plot(self.x, self.a, color="blue", label=r'$\alpha_t$', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.plot(self.x, lowerboundCI, color="red", label='90% Confidence Interval', linewidth=0.5, linestyle="dashed")
        plt.plot(self.x, upperboundCI, color="red", linewidth=0.5, linestyle="dashed")
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state ' + r'$\alpha_t$ ' + 'and its 90% confidence intervals',fontsize=12)
        plt.draw()

        """ 2.1.ii """
        plt.figure()
        plt.plot(self.x, self.P, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Filtered state variance ' + r'$P_t$',fontsize=12)
        plt.draw()

        """ 2.1.iii """
        self.v[0] = 0
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.v, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Prediction errors ' + r'$v_t$',fontsize=12)
        plt.draw()

        """ 2.1.iv """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.ylim(bottom=20000, top=32500)
        plt.plot(self.x, self.f, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Prediction errors ' + r'$F_t$',fontsize=12)
        plt.draw()

    def fig2(self):
        """ 2.2.i """
        lowerboundCI = np.zeros(self.n)
        upperboundCI = np.zeros(self.n)
        for i in range(self.n):
            lowerboundCI[i] = self.alphahat[i] - 1.645*np.sqrt(self.V[i])
            upperboundCI[i] = self.alphahat[i] + 1.645*np.sqrt(self.V[i])

        plt.figure()
        plt.plot(self.x, self.alphahat, color="blue", label='Smoothed state', linewidth=0.5)
        plt.plot(self.x, self.y, color="grey", label='Nile data', linewidth=1, alpha=0.3)
        plt.plot(self.x, lowerboundCI, color="red", label='90% Confidence Interval', linewidth=0.5, linestyle="dashed")
        plt.plot(self.x, upperboundCI, color="red", linewidth=0.5, linestyle="dashed")
        plt.xticks(np.arange(10), self.xYears)
        plt.legend(loc='upper right')
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Nile data and output of state smoothing recursion',fontsize=12)
        plt.draw()

        """ 2.2.ii """
        plt.figure()
        plt.ylim(bottom=2000, top=4200)
        plt.plot(self.x, self.V, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothed state variance ' + r'$V_t$',fontsize=12)
        plt.draw()

        """ 2.2.iii """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.r, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothing cumulant ' + r'$r_t$',fontsize=12)
        plt.draw()

        """ 2.2.iv """
        plt.figure()
        plt.plot(self.x, self.N, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Smoothing variance cumulant ' + r'$N_t$',fontsize=12)
        plt.draw()

    def fig3(self):
        """ 2.3.i """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.epshat, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Observation error ' + r'$\hat{\epsilon}_t$',fontsize=12)
        plt.draw()

        """ 2.3.ii """
        plt.figure()
        plt.plot(self.x, self.varepshatgiveny, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Observation error variance ' + r'$Var(\epsilon_t | y)$',fontsize=12)
        plt.draw()

        """ 2.3.iii """
        plt.figure()
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.plot(self.x, self.etahat, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('State error ' + r'$\hat{\eta}_t$',fontsize=12)
        plt.draw()

        """ 2.3.iv """
        plt.figure()
        plt.plot(self.x, self.varetagiveny, color="blue", linewidth=0.5)
        plt.xticks(np.arange(10), self.xYears)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('State error variance ' + r'$Var(\eta_t | y)$',fontsize=12)
        plt.draw()

def main():
    nile = Nile()
    nile.KalmanFilter()
    nile.KalmanSmoother()
    nile.DisturbanceSmoother()
    nile.fig1()
    nile.fig2()
    nile.fig3()
    plt.show()

if __name__ == "__main__":
    main()
