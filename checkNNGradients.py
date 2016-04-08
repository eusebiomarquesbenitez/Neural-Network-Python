#!/usr/bin/env python
# coding: utf-8

import numpy as np
import NN


def debugInitializeWeights(len_out, len_in):
    """
    W = DEBUGINITIALIZEWEIGHTS(len_out, len_in) initializes the weights
    of a layer with len_in incoming connections and len_out outgoing
    connections using a fix set of values.
    """
    W = np.zeros((len_out, len_in + 1))
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    c, r = W.shape
    W = np.reshape(np.sin(range(1, W.size+1)), (r, c)).T / 10
    return W


def computeNumericalGradient(J, W):
    """
    COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    and gives us a numerical estimate of the gradient.
    """
    numgrad = np.zeros(W.shape)
    perturb = np.zeros(W.shape)
    epsilon = 1e-4
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            perturb[i][j] = epsilon
            loss1 = J(W - perturb)[0]
            loss2 = J(W + perturb)[0]
            numgrad[i][j] = (loss2 - loss1) / (2 * epsilon)
            perturb[i][j] = 0
    return numgrad


def checkNNGradients(learn_rate):
    """
    checkNNGradients(learn_rate) Creates a small neural network to check the
    backpropagation gradients, it will output the analytical gradients produced
    by your backprop code and the numberical gradients (computed using compute-
    NumericalGradient). These two gradient computations should result in very
    similar values.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some 'random' test data
    W1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    W2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Generate X, y
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.array([i % num_labels for i in range(1, m + 1)])

    # Unroll parameters
    W = np.hstack((W1.flatten(0), W2.flatten(0)))
    W = W.reshape((len(W), 1))

    def costFunc(p):
        return NN.nnCostFunction(p, input_layer_size, hidden_layer_size,
                                 num_labels, X, y, learn_rate)

    cost, grad = costFunc(W)
    numgrad = computeNumericalGradient(costFunc, W)

    for i in range(len(grad)):
        print "%10f\t%10f" % (grad[i], numgrad[i])
    print "The above two lines you get should be very similar.\n"

    diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
    print ("If your backpropagation implementation is correct, then"
           "\nthe relative difference will be small (less than 1e-9).\n"
           "\nRelative Difference: %g\n") % diff


if __name__ == '__main__':
    checkNNGradients(0)
