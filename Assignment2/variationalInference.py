import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

muTrue = 1
lambdaTrue = 4
aTrue = 30
bTrue = 10
precisionTrue = aTrue / bTrue  # tau

NUM_DATAPOINTS = 400
NUM_ITERATIONS = 10000

'''
def sampleGaussian(mean, precision):
    return np.random.normal(mean, np.sqrt(precision ** -1))


def sampleGamma(alpha, beta):
    return np.random.gamma(alpha, 1 / beta)
'''


def generateData():
    return np.random.normal(muTrue, np.sqrt(precisionTrue ** (-1)), NUM_DATAPOINTS)


# p(D | mu, tau)
def likelihood(X, mu, tau):
    N = NUM_DATAPOINTS
    return ((tau / (2 * np.pi)) ** (N / 2)) * np.exp((-tau / 2) * np.sum(np.square(X - mu)))


'''
def muExpectationExpression(X, b0, lambda0, mu0):
    muExpected = np.average(X)
    tauExpected = 1 / ((1 / (NUM_DATAPOINTS - 1)) * np.sum(np.square(X - muExpected)))
    muExpected2 = (muExpected ** 2) + 1 / (NUM_DATAPOINTS * tauExpected)
    return b0 + lambda0 * (muExpected2 + (mu0 * mu0) - 2 * muExpected * mu0) \
           + 0.5 * np.sum(np.square(X) + muExpected2 - 2 * muExpected * X)
'''


# In fact constant over iterations
def iterateA(a0):
    return a0 + ((NUM_DATAPOINTS + 1) / 2.0)


# In fact constant over iterations
def iterateMu(X, lambda0, mu0):
    return ((lambda0 * mu0) + (NUM_DATAPOINTS * np.mean(X))) / (lambda0 + NUM_DATAPOINTS)


def iterateB(X, b0, lambda0, mu0, muApprox, lambdaApprox):
    '''
    muExpected = np.average(X)
    tauExpected = 1 / ((1 / (NUM_DATAPOINTS - 1)) * np.sum(np.square(X - muExpected)))
    muExpected2 = (muExpected ** 2) + 1 / (NUM_DATAPOINTS * tauExpected)
    '''

    muExpected = muApprox
    muExpected2 = (1.0 / lambdaApprox) + (muApprox ** 2)

    return b0 + lambda0 * (muExpected2 + (mu0 * mu0) - (2 * muExpected * mu0)) \
           + 0.5 * np.sum(np.square(X) + muExpected2 - 2 * muExpected * X)

    # return b0 + 0.5 * muExpectationExpression(X, b0, lambda0, mu0)


# expected tau is a/b (standard result for mean of gamma dist. provided in Bishop)
def iterateLambda(lambda0, aApprox, bApprox):
    return (lambda0 + NUM_DATAPOINTS) * (aApprox / bApprox)


def variationalInf():
    mu0, lambda0, a0, b0 = 0, 1, 0, 1

    # mu = mu0
    lam = lambda0
    # a = a0
    b = b0

    # Constants throughout iterations
    a = iterateA(a0)
    mu = iterateMu(X, lambda0, mu0)

    for i in range(NUM_ITERATIONS):
        b = iterateB(X, b0, lambda0, mu0, mu, lam)
        lam = iterateLambda(lambda0, a, b)

    return mu, lam, a, b


def qMu(x, mean, precision):
    return stats.norm.pdf(x, mean, np.sqrt(1.0 / precision))


def qTau(tau, alpha, beta):
    return stats.gamma.pdf(tau, alpha, loc=0, scale=(1.0 / beta))


def plotContours(mean, precision, alpha, beta, X):
    muVals = np.linspace(muTrue - 0.5, muTrue + 0.5, 100)
    tauVals = np.linspace(precisionTrue - 0.75, precisionTrue + 0.75, 100)
    M, T = np.meshgrid(muVals, tauVals, indexing="ij")
    Z = np.zeros_like(M)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = qMu(muVals[i], mean, precision) * qTau(tauVals[j], alpha, beta)

    muValsTrue = np.linspace(muTrue - 0.5, muTrue + 0.5, 100)
    tauValsTrue = np.linspace(precisionTrue - 0.75, precisionTrue + 0.75, 100)
    Mtrue, Ttrue = np.meshgrid(muValsTrue, tauValsTrue, indexing="ij")
    Ztrue = np.zeros_like(Mtrue)

    for i in range(Ztrue.shape[0]):
        for j in range(Ztrue.shape[1]):
            Ztrue[i][j] = qMu(muValsTrue[i], muTrue, lambdaTrue * tauValsTrue[j]) * qTau(tauValsTrue[j], aTrue, bTrue) * likelihood(X, muValsTrue[i], tauValsTrue[j])

    '''
    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="green", lw=4)]
    '''
    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="green", lw=4)]

    fig, ax = plt.subplots()
    ax.legend(custom_lines, ['Approximated', 'True Posterior'])

    # fig, ax = plt.subplots()
    plt.contour(M, T, Z, 5, colors="red")
    plt.contour(Mtrue, Ttrue, Ztrue, 5, colors="green")
    plt.xlabel("Mean")
    plt.ylabel("Precision")
    # ax.legend(custom_lines, ['Approximated', 'True'])
    # plt.legend()
    plt.title("Approximated Posterior vs True Posterior\nN: " + str(NUM_DATAPOINTS) + ", "
              "Iterations: " + str(NUM_ITERATIONS) + "\n(mu, lambda, a, b) = ("
              + str(muTrue) + ", " + str(lambdaTrue) + ", " + str(aTrue) + ", " + str(bTrue) + ")")
    plt.show()


X = generateData()
print("Likelihood:", likelihood(X, muTrue, 1))
muApp, lamApp, aApp, bApp = variationalInf()

print("Approximated values:\n", muApp, lamApp, aApp, bApp)
print("Real values:\n", muTrue, lambdaTrue, aTrue, bTrue)

plotContours(muApp, lamApp, aApp, bApp, X)
