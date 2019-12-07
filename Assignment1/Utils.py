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


def plotCurves(xVals, yValsArr):
    # x = np.arange(len(yValsArr[0]))
    for yVals in yValsArr:
        plt.plot(xVals, yVals)
    plt.show()


def plotCurvesWithPoints(xVals, yValsArr, pointsX, pointsY, variance=None):
    assert len(pointsX) == len(pointsY)
    for yVals in yValsArr:
        plt.plot(xVals, yVals, zorder=0)

    for i in range(len(pointsX)):
        # print("x: ", pointsX[i], ", y: ", pointsY[i])
        plt.scatter(pointsX, pointsY, zorder=1, color='black')

    if variance is not None:
        mean = yValsArr[0]
        print("Var: ", variance)
        plt.fill_between(xVals, mean + 2 * np.sqrt(variance), mean - 2 * np.sqrt(variance), color="black", alpha=0.2)

    # plt.title("GP Posterior")
    # plt.title("GP Posterior, Predictive Certainty")
    plt.title("GP Posterior, Predictive Certainty, Noise")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plotLines(lines):
    x = np.arange(10)
    for line in lines:
        plt.plot(x, line[1] * x + line[0])
    plt.show()
