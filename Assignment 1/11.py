import numpy as np
import math
import matplotlib.pyplot as plt

class LocalLevel:
    def __init__(self):
        self.a1 = 0
        self.p1 = 10 ** 7
        self.vareps = 1
        self.vareta = [10, 1, 0.1, 0.001]

        self.n = 100

        # how far you want to go to the past
        self.past = 20

    def mu(self):
        mu1 = np.random.normal(self.a1, math.sqrt(self.p1))
        mu = np.zeros(self.n)
        mu[0] = mu1

        return mu

    def sim(self, vareta):

        np.random.seed(1)

        y = np.zeros(self.n)
        mu = self.mu()

        eps = np.zeros(self.n)
        eta = np.zeros(self.n)

        for i in range(self.n):
            eps[i] = np.random.normal(0, math.sqrt(self.vareps))
            eta[i] = np.random.normal(0, math.sqrt(vareta))

        y[0] = mu[0] + eps[0]

        for t in range(1, self.n):
            mu[t] = mu[t-1] + eta[t]
            y[t] = mu[t] + eps[t]

        return y, mu

    def weightFunction(self, t, vareta):
        weight = np.zeros(self.past)

        for j in range(self.past):
            weight[self.past - 1 - j] = self.k[t - j - 1]
            for m in range(j):
                weight[self.past - 1 - j] *= (1 - self.k[t - m - 1])

        return weight

    def kalmanFilter(self, y, vareta):
        """ PART 2 KALMAN FILTER """
        v = np.zeros(self.n)
        f = np.zeros(self.n)
        self.k = np.zeros(self.n)

        a = np.zeros(self.n + 1)
        a[0] = self.a1
        P = np.zeros(self.n + 1)
        P[0] = self.p1

        for t in range(self.n):
            v[t] = y[t] - a[t]
            f[t] = P[t] + self.vareps
            self.k[t] = P[t]/f[t]
            a[t+1] = a[t] + self.k[t]*v[t]
            P[t+1] = self.k[t]*self.vareps + vareta

        return a[1:], P[1:]

    def a_to_d(self):
        sims = len(self.vareta)

        y = np.zeros((sims, self.n))
        mu = np.zeros((sims, self.n))

        estimate = np.zeros((sims, self.n))
        Pest = np.zeros((sims, self.n))

        weights = np.zeros((sims, self.past))

        for i in range(sims):
            """ PART 1 SIMULATE A TIME SERIES """
            y[i], mu[i] = self.sim(self.vareta[i])

            """ PART 2 KALMAN FILTER """
            estimate[i], Pest[i] = self.kalmanFilter(y[i], self.vareta[i])

            """ PART 3 PLOT OBSERVATIONS AGAINST FILTERED ESTIMATES """
            plt.figure()
            plt.plot(np.linspace(0,self.n,self.n), y[i], color="blue", label="observations", linewidth=0.5)
            plt.plot(np.linspace(0,self.n,self.n), estimate[i], color="red", label="Filtered estimates", linewidth=0.5)
            # plt.plot(np.linspace(0,self.n,self.n), mu[i], color="red",linewidth=1)
            plt.legend(loc='upper left')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(r'$t$',fontsize=16)
            plt.title('Simulation with q = ' + str(self.vareta[i]/self.vareps),fontsize=18)
            plt.grid(True,linewidth=1.5,linestyle=':')
            plt.subplots_adjust(left=0.15,bottom=0.15)
            plt.draw()

            """ PART 4 PRESENT WEIGHT FUNCTIONS """
            weights[i] = self.weightFunction(self.n, self.vareta[i])

        """ CONTINUE PART 4: PLOT WEIGHT FUNCTIONS """
        xbars = []
        for i in range(self.past):
            pastYear = i - self.past + 1
            xbars.append(pastYear)

        xbars = tuple(xbars)
        yPos = np.arange(len(xbars))

        plt.figure()
        weightplot = plt.subplot(111)

        width = 0.2

        for i in range(sims):
            plt.bar(yPos + 0.5*sims*width - i*(1.25*width), weights[i], width=width, label='q = ' + str(self.vareta[i]/self.vareps))

        plt.xticks(yPos, xbars)
        plt.legend(loc = 'upper left')
        plt.title('Weight functions',fontsize=18)
        plt.subplots_adjust(left=0.15,bottom=0.15)
        plt.draw()

    def kalmanFilterE(self, y, vareta):
        """ PART 2 KALMAN FILTER """
        v = np.zeros(self.n)
        f = np.zeros(self.n)
        self.k = np.zeros(self.n)

        a = np.zeros(self.n + 1)
        a[0] = self.a1
        a[1] = y[0]
        P = np.zeros(self.n + 1)
        P[0] = self.p1
        P[1] = self.vareps + vareta

        for t in range(1, self.n):
            v[t] = y[t] - a[t]
            f[t] = P[t] + self.vareps
            self.k[t] = P[t]/f[t]
            a[t+1] = a[t] + self.k[t]*v[t]
            P[t+1] = self.k[t]*self.vareps + vareta

        return a[1:], P[1:]

    def e(self):
        vareta = 0.1

        y = np.zeros(self.n)
        mu = np.zeros(self.n)

        estimate = np.zeros(self.n)
        Pest = np.zeros(self.n)

        """ PART 1 SIMULATE A TIME SERIES """
        y, mu = self.sim(vareta)

        """ PART 2 KALMAN FILTER """
        estimate, Pest = self.kalmanFilterE(y, vareta)

        """ PART 3 PLOT OBSERVATIONS AGAINST FILTERED ESTIMATES """
        plt.figure()
        plt.plot(np.linspace(0,self.n,self.n), y, color="blue", label="observations", linewidth=0.5)
        plt.plot(np.linspace(0,self.n,self.n), estimate, color="red", label="Filtered estimates", linewidth=0.5)
        # plt.plot(np.linspace(0,self.n,self.n), mu, color="red",linewidth=1)
        plt.legend(loc='upper left')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$t$',fontsize=16)
        plt.title('Simulation with q = ' + str(vareta/self.vareps),fontsize=18)
        plt.grid(True,linewidth=1.5,linestyle=':')
        plt.subplots_adjust(left=0.15,bottom=0.15)
        plt.draw()

        """ PART 4 PRESENT WEIGHT FUNCTIONS """
        weight = self.weightFunction(self.n, vareta)

        xbars = []
        for i in range(self.past):
            pastYear = i - self.past + 1
            xbars.append(pastYear)

        xbars = tuple(xbars)
        yPos = np.arange(len(xbars))

        plt.figure()
        width = 0.2
        plt.bar(yPos, weight, width=width, label='q = ' + str(vareta/self.vareps))
        plt.xticks(yPos, xbars)
        plt.legend(loc = 'upper left')
        plt.title('Weight functions',fontsize=18)
        plt.subplots_adjust(left=0.15,bottom=0.15)
        plt.draw()

def main():
    localLevel = LocalLevel()

    localLevel.a_to_d()
    localLevel.e()
    plt.show()

if __name__ == "__main__":
    main()
