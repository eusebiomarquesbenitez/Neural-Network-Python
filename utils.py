#!/usr/bin/env python
# coding:utf-8


import numpy as np


def loadData(file_X, file_y):
    '''
    input: filepath of X, y
    output: X, y in numpy format
    '''
    X = np.loadtxt(file_X)
    y = np.loadtxt(file_y)
    return X, y


def loadParams(file_w1, file_w2):
    '''
    input: filepath of parameters
    output: parameters in numpy format
    '''
    W1 = np.loadtxt(file_w1)
    W2 = np.loadtxt(file_w2)
    return W1, W2


def randInitializeWeights(len_in, len_out):
    INIT_EPSILON = 0.12
    return np.random.uniform(low=-INIT_EPSILON, high=INIT_EPSILON,
                             size=(len_out, len_in)).astype(np.float64)


if __name__ == '__main__':
#    fx = 'X.txt'
#    fy = 'y.txt'
#    X, y = loadData(fx, fy)
#    print X
#    print y
#    print X[0]
#    print y[0]
#    print X.shape
#    print y.shape
#    fx = 'w1.txt'
#    fy = 'w2.txt'
#    X, y = loadData(fx, fy)
##    print X
##    print y
##    print X[0]
##    print y[0]
#    print X.shape
#    print y.shape
#    z = X.flatten(1)
#    print z.shape
#    print type(z)
#    print z
#    z = z.reshape((len(z),1))
#    print z.shape
#    print z

    w = debugInitializeWeights(3, 4)
    print w
    print w.shape
