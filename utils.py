# coding:utf-8

import numpy as np


def loadData(file_X, file_y):
    X = np.loadtxt(file_X)
    y = np.loadtxt(file_y)
    return X, y


def loadParams(file_W1, file_W2):
    W1 = np.loadtxt(file_W1)
    W2 = np.loadtxt(file_W2)
    return W1, W2


def randInitializeWeights(len_in, len_out):
    INIT_EPSILON = 0.12
    return np.random.uniform(low=-INIT_EPSILON, high=INIT_EPSILON,
                             size=(len_out, len_in + 1)).astype(np.float64)
