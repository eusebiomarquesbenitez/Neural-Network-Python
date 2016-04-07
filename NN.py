# coding: utf-8

import numpy as np
from activation_function import Sigmoid


def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y,
                   learn_rate):
    W1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))
    W2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                    (num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]

    # PART 3
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

    # Unregularized cost function
    sumj = 0
    for i in range(m):
        sumj += ny[i, :].dot(np.log(hx[i, :].T)) + (1-ny[i,:]).dot(np.log(1 - hx[i, :].T))
    J = -1.0 / m * sumj

    # PART 4
    # Regularized cost function
    w1_sqrt = W1**2
    w2_sqrt = W2**2
    sum_w = sum(np.sum(w1_sqrt[:,1:],0)) + sum(np.sum(w2_sqrt[:,1:],0))
    J += learn_rate/(2.0 * m) * sum_w

    # PART 7
    # Backpropagation
    delta3 = a3 - ny # 5000 x 10
    delta2 = delta3.dot(W2[:,1:]) * Sigmoid.dy_dz(Sigmoid.y(z2))
    Delta1 = delta2.T.dot(a1)
    Delta2 = delta3.T.dot(a2)
    W1_grad = Delta1 / m
    W2_grad = Delta2 / m

    # PART 8
    # Implement Regularization
    tmp_W1 = (1.0 * learn_rate / m) * W1
    tmp_W2 = (1.0 * learn_rate / m) * W2
    tmp_W1[:, 0] = 0
    tmp_W2[:, 0] = 0
    W1_grad += tmp_W1
    W2_grad += tmp_W2

    # Unroll gradients
    W_grad = np.hstack((W1_grad.flatten(0), W2_grad.flatten(0)))
    W_grad = W_grad.reshape((len(W_grad), 1))
    return J, W_grad
