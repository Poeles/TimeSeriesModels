import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class LocalLevel:
    def __init__(self):

        np.random.seed(0)

        self.a1 = 0
        self.p1 = 10 ** 7
        self.vareps = 1
        self.vareta = float(input("Pick a value for q (10, 1, 0.1, 0.001). "))
        self.Tsize = int(input("Choose the length for the time series(50, 100, 200). "))

        self.n = 100

        self.data = [i.strip().split() for i in open("DK-data/Nile.dat").readlines()]
        self.data.pop(0)

    def mu(self):
        mu1 = np.random.normal(self.a1, math.sqrt(self.p1))
        mu = np.zeros((self.n, self.Tsize))

        for i in range(self.n):
            mu[i][0] = mu1

        return mu

    def sim(self):

        self.y = np.zeros((self.n, self.Tsize))
        mu = self.mu()

        eps = np.zeros((self.n, self.Tsize))
        eta = np.zeros((self.n, self.Tsize))

        for i in range(self.n):
            for t in range(self.Tsize):
                eps[i][t] = np.random.normal(0, math.sqrt(self.vareps))
                eta[i][t] = np.random.normal(0, math.sqrt(self.vareta))

            self.y[i][0] = mu[i][0] + eps[i][0]

            for t in range(1, self.Tsize):
                mu[i][t] = mu[i][t-1] + eta[i][t]
                self.y[i][t] = mu[i][t] + eps[i][t]

        return self.y, mu

    def Ldc(self, y, q):
        v = [0 for t in range(self.Tsize)]
        fstar = [0 for t in range(self.Tsize)]
        k = [0 for t in range(self.Tsize)]
        a = [0 for t in range(self.Tsize + 1)]
        Pstar = [0 for t in range(self.Tsize + 1)]

        # diffuse prior density
        a[0] = 0
        Pstar[0] = 10**7

        a[1] = y[0]
        Pstar[1] = 1 + q

        #Kalman filter
        for t in range(1, self.Tsize):
            v[t] = y[t] - a[t]
            fstar[t] = Pstar[t] + 1
            a[t+1] = a[t] + Pstar[t]/fstar[t]*v[t]
            Pstar[t+1] = Pstar[t]*(1 - Pstar[t]/fstar[t]) + q

        varepshat = 0
        for t in range(1, self.Tsize):
            varepshat += (v[t]**2) / fstar[t]

        varepshat = varepshat/(self.Tsize-1)

        Ldc = -(self.Tsize/2)*np.log(2*np.pi) - (self.Tsize - 1)/2 - ((self.Tsize - 1)/2)*np.log(varepshat)

        for t in range(1, self.Tsize):
            Ldc += -(1/2)*np.log(fstar[t])

        return Ldc

    def varepshat(self, y, q):
        v = [0 for t in range(self.Tsize)]
        fstar = [0 for t in range(self.Tsize)]
        k = [0 for t in range(self.Tsize)]
        a = [0 for t in range(self.Tsize + 1)]
        Pstar = [0 for t in range(self.Tsize + 1)]

        # diffuse prior density
        a[0] = 0
        Pstar[0] = 10**7

        a[1] = y[0]
        Pstar[1] = 1 + q

        #Kalman filter
        for t in range(1, self.Tsize):
            v[t] = y[t] - a[t]
            fstar[t] = Pstar[t] + 1
            a[t+1] = a[t] + Pstar[t]/fstar[t]*v[t]
            Pstar[t+1] = Pstar[t]*(1 - Pstar[t]/fstar[t]) + q

        varepshat = 0
        for t in range(1, self.Tsize):
            varepshat += (v[t]**2) / fstar[t]

        varepshat = varepshat/(self.Tsize-1)

        return varepshat


def main():
    localLevel = LocalLevel()

    y, mu = localLevel.sim()

    # vareps, q
    varepshat = [0 for i in range(localLevel.n)]
    varetahat = [0 for i in range(localLevel.n)]
    qhat = [0 for i in range(localLevel.n)]
    bnd = ((1e-6, 10),)

    for i in range(localLevel.n):
        qhat[i] = minimize(lambda q: -localLevel.Ldc(y[i], q), 0.0001, bounds=bnd).x[0]
        varepshat[i] = localLevel.varepshat(y[i], qhat[i])
        varetahat[i] = qhat[i]*varepshat[i]

        print("varepshat: {}, varetahat: {}".format(varepshat[i], varetahat[i]))


if __name__ == "__main__":
    main()
