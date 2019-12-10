import numpy as np
import scipy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

N = 200
D = 10
A = np.random.normal(0, 1, [10, 2])
global S


def flin(x):
    return x @ np.transpose(A)


def fnonlin(x):
    return np.transpose(np.array([np.sin(x) - x * np.cos(x), np.cos(x) + x * np.sin(x)]))


def f(x):
    x = np.reshape(x, (10, 2))
    wwt = x @ np.transpose(x)
    cInv = np.linalg.inv(wwt)
    print("S:", S)
    val = (N / 2) * (D * np.log(2 * np.pi) + np.log(np.linalg.det(wwt) + np.trace(cInv * S)))
    return val


def dfx(x):
    return 0


def generateDataRep():
    x = np.linspace(0, 4 * np.pi, N)
    Y = flin(fnonlin(x))
    return x, Y


def calcS(Y):
    yMean = np.mean(Y, axis=0)  # .reshape(-1, 1)
    S = np.zeros((len(yMean), len(yMean)))
    for i in range(N):
        diffVector = (Y[i] - yMean).reshape(-1, 1)
        S += diffVector @ np.transpose(diffVector)
    S /= N
    return S


x, Y = generateDataRep()
x0 = np.random.randn(len(Y[0]) * 2)

S = calcS(Y)
# Y = np.transpose(Y)
# print(Y.shape)
# print(np.shape(S))
# print(S)
x_star = opt.fmin_cg(f, x0, fprime=dfx)
