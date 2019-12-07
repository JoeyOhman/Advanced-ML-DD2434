import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from Utils import plotBivar, plotLines, plotCurves, plotCurvesWithPoints

NUM_DATAPOINTS = 501  # 201
SIGMA_GEN = 1
SIGMA = 1
TAU = 1


def plotPrior():
    muPrior = [0, 0]
    plotBivar(muPrior, [[TAU, 0], [0, TAU]])


def generateData():
    genW = [0.5, -1.5]  # np.random.normal(0, 1, 2)
    x = np.linspace(-1, 1, NUM_DATAPOINTS)
    # x = np.array([[1, val] for val in x])
    t = genW[0] * np.transpose(x)[1] + genW[1] + np.random.normal(0, SIGMA_GEN, NUM_DATAPOINTS)
    return x, t


def generateData2():
    x = np.array([-4, -3, -2, -1, 0, 2, 3, 5])
    t = 2 + ((0.5 * x - 1) ** 2) * np.sin(3 * x) + np.random.normal(0, np.sqrt(3), len(x))
    return x, t


# sigma is likelihood variance in t
# tau is prior variance in weights
# x, t are vectors
# posterior over W
def findPosterior(x, t, sigma, tau):
    sigmaSquareInverse = (1 / (sigma ** 2))
    xTranspose = np.transpose(x)
    # print(np.transpose(x) @ x)
    # print(sigmaSquareInverse * (np.transpose(x) @ x) + 1 / (tau ** 2))
    sigmaMatrixInverse = np.linalg.inv(sigmaSquareInverse * (xTranspose @ x)
                                       + (1 / (tau ** 2)) * np.identity(2))

    mu = sigmaSquareInverse * sigmaMatrixInverse @ xTranspose @ t
    return mu, sigmaMatrixInverse


def kernel(x1, x2, sigma, l):
    k = []
    for i in range(len(x1)):
        diff = (x1[i] - x2)
        exponent = diff * diff / (l * l)
        newRow = sigma * np.exp(- exponent)
        k.append(newRow)

    return np.array(k)


def priorGP(x, sigma, l):
    k = kernel(x, x, sigma, l)
    return np.zeros(len(x)), k


def posteriorGP(x, xPredict, f, sigma, l):
    invKernelX = np.linalg.inv(kernel(x, x, sigma, l) + np.identity(len(x)) * 0.05)
    mean = kernel(xPredict, x, sigma, l) @ invKernelX @ f.reshape(-1, 1)
    cov = kernel(xPredict, xPredict, sigma, l) - \
          kernel(xPredict, x, sigma, l) @ invKernelX @ kernel(x, xPredict, sigma, l)
    # for i in range(len(cov)):
        # cov[i][i] += 0.05
    return mean, cov


# plotPrior()
x, t = generateData2()
xPredict = np.linspace(-10, 10, NUM_DATAPOINTS)
x = x[0:8]
t = t[0:8]
# muGP, kGP = priorGP(x, 1, 0.5)
# sample = np.random.multivariate_normal(muGP, kGP, 2)
muPost, kPost = posteriorGP(x, xPredict, t, 1, 1)
# print(np.round(muPost, 3))
# print(np.round(kPost, 3))
sample = np.random.multivariate_normal(np.transpose(muPost)[0], kPost, 10)
# plotCurves(xPredict, [sample])
# plotCurvesWithPoints(xPredict, sample, x, t)
plotCurvesWithPoints(xPredict, np.transpose(muPost), x, t, variance=np.diag(kPost))
'''
mu, sigma = findPosterior(x, t.reshape(-1, 1), SIGMA, TAU)
muRow = [mu[0][0], mu[1][0]]
plotBivar(muRow, sigma)
functions = np.random.multivariate_normal(muRow, sigma, 5)
plotLines(functions)
'''
