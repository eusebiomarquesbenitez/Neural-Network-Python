# coding: utf-8

import numpy as np
from activation_function import Sigmoid


def nnCostFunction(W1, W2,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y,
                   learn_rate):
#    W1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
#                    (hidden_layer_size, (input_layer_size + 1)))
#    W2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
#                    (num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]

    # PART 1
    # Feedforward Computation
    X = np.hstack((np.ones((m, 1)), X))
    a1 = X #5000x401
    z2 = a1.dot(W1.T) #5000 x 25
    a2 = Sigmoid.y(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2)) #5000 x 26
    z3 = a2.dot(W2.T) #5000 x 10
    a3 = Sigmoid.y(z3)
    hx = a3 # 5000 x 10

    ny = np.zeros((y.shape[0], num_labels)) # 5000 x 10
    for i in range(y.shape[0]):
        ny[i, y[i] - 1] = 1

    sumj = 0
    for i in range(m):
        sumj += ny[i, :].dot(np.log(hx[i, :].T)) + (1-ny[i,:]).dot(np.log(1 - hx[i, :].T))
    J = -1.0 / m * sumj
    return J
