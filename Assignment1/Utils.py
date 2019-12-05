import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotBivar(mu, cov):
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = stats.multivariate_normal(mu, cov)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
    ax.set_xlabel('w0')
    ax.set_ylabel('w1')
    ax.set_zlabel('p(w0, w1)')
    plt.show()


def plotCurves(yVals):
    x = np.arange(len(yVals[0]))
    for y in yVals:
        plt.plot(x, y)
    plt.show()


def plotLines(lines):
    x = np.arange(10)
    for line in lines:
        plt.plot(x, line[1] * x + line[0])
    plt.show()
