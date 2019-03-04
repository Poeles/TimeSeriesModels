import numpy as np
import math
from scipy.stats import gaussian_kde
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.optimize import minimize
import seaborn as sns
import statsmodels.api as sm
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

        self.x = np.linspace(0,10,self.n)

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

    def histogram(self, vareps, vareta):
        plt.figure()
        plt.hist(vareps, 20, rwidth=0.85)
        plt.title('Histogram for ' + r'$\hat{\sigma}_{\varepsilon}^{2}$ ' + \
        ' with ' + r'$\sigma_{\varepsilon}^{2}$ = ' + str(self.vareps) + ', T = ' + str(self.Tsize),fontsize=12)
        plt.draw()

        plt.figure()
        plt.hist(vareta, 20, rwidth=0.85)
        plt.title('Histogram for ' + r'$\hat{\sigma}_{\eta}^{2}$ ' + \
        ' with ' + r'$\sigma_{\eta}^{2}$ = ' + str(self.vareta) + ', T = ' + str(self.Tsize),fontsize=12)
        plt.draw()

    def bias(self, vareps, vareta):

        for i in range(self.n):
            vareps[i] = vareps[i] - self.vareps
            vareta[i] = vareta[i] - self.vareta

        plt.figure()
        plt.ylim(bottom=-self.vareps, top=self.vareps)
        plt.plot(self.x, vareps, color="blue", linewidth=0.5)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.xlabel(r'$i$',fontsize=16)
        plt.title('Bias of ' + r'$\hat{\sigma}_{\varepsilon}^{2}$' + \
        ' with ' + r'$\sigma_{\varepsilon}^{2}$ = ' + str(self.vareps) + ', T = ' + str(self.Tsize),fontsize=12)
        plt.draw()

        plt.figure()
        plt.ylim(bottom=-self.vareta, top=10*self.vareta)
        plt.plot(self.x, vareta, color="blue", linewidth=0.5)
        plt.axhline(y=0, color='k', linewidth=0.5)
        plt.xlabel(r'$i$',fontsize=16)
        plt.title('Bias of ' + r'$\hat{\sigma}_{\eta}^{2}$ ' + \
        ' with ' + r'$\sigma_{\eta}^{2}$ = ' + str(self.vareta) + ', T = ' + str(self.Tsize),fontsize=12)
        plt.draw()

    def samplevariance(self, vareps, vareta):
        samplevareps = np.var(vareps, ddof=1)
        samplevareta = np.var(vareta, ddof=1)

        print("Sample variance of vareps: {}".format(samplevareps))
        print("Sample variance of vareta: {}".format(samplevareta))

    def skewnessKurtosis(self, vareps, vareta):
        print('Skewness of vareps: {}'.format(skew(vareps)))
        print('Skewness of vareta: {}'.format(skew(vareta)))
        print('Kurtosis of vareps: {}'.format(kurtosis(vareps)))
        print('Kurtosis of vareta: {}'.format(kurtosis(vareta)))


def main():
    localLevel = LocalLevel()

    y, mu = localLevel.sim()

    # vareps, q
    varepshat = np.zeros(localLevel.n)
    varetahat = np.zeros(localLevel.n)
    qhat = np.zeros(localLevel.n)
    bnd = ((1e-6, None),)

    for i in range(localLevel.n):
        qhat[i] = minimize(lambda q: -localLevel.Ldc(y[i], q), 1e-6, bounds=bnd).x[0]
        varepshat[i] = localLevel.varepshat(y[i], qhat[i])
        varetahat[i] = qhat[i]*varepshat[i]

    print("AVG vareps: {}, AVG vareta: {}".format(np.mean(varepshat), np.mean(varetahat)))

    localLevel.histogram(varepshat, varetahat)
    #localLevel.bias(varepshat, varetahat)
    #localLevel.samplevariance(varepshat, varetahat)
    #localLevel.skewnessKurtosis(varepshat, varetahat)
    plt.show()


if __name__ == "__main__":
    main()
