#!/usr/bin/env python
# coding:utf-8

import numpy as np
import conf
import NN
from activation_function import Sigmoid
import utils


if __name__ == '__main__':
    print "Part 1: Loading Data\n"

    X, y = utils.loadData(conf.FILE_X, conf.FILE_Y)

    print "Part 2: Loading Parameters\n"

    W1, W2 = utils.loadParams(conf.FILE_W1, conf.FILE_W2)
    # Unroll parameters
    W = np.hstack((W1.flatten(0), W2.flatten(0)))
    W = W.reshape((len(W), 1))

    print "Part 3: Compute Cost(Feedforward)\n"

    LEARN_RATE = 0
    J, _ = NN.nnCostFunction(W, conf.INPUT_LAYER_SIZE, conf.HIDDEN_LAYER_SIZE,
                             conf.NUM_LABELS, X, y, LEARN_RATE)
    print ("Cost at parameters (loaded from w1.txt and w2.txt): %f"
           "\n(this value should be about 0.287629)\n") % J

    print "Part 4: Implement Regularization\n"

    LEARN_RATE = 1
    J, _ = NN.nnCostFunction(W, conf.INPUT_LAYER_SIZE, conf.HIDDEN_LAYER_SIZE,
                             conf.NUM_LABELS, X, y, LEARN_RATE)
    print ("Cost at parameters (loaded from w1.txt and w2.txt): %f"
           "\n(this value should be about 0.383770)\n") % J

    print "Part 5: Sigmoid Gradient\n"

    g = Sigmoid.dy_dz(Sigmoid.y(np.array([-1, -0.5, 0, 0.5, 1])))
    print "Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:", g
    print "(this shoulde be [0.1966 0.2350 0.2500 0.2350 0.1966])\n"

    print "Part 6: Initializing Parameters\n"

    init_W1 = utils.randInitializeWeights(conf.INPUT_LAYER_SIZE,
                                          conf.HIDDEN_LAYER_SIZE)
    init_W2 = utils.randInitializeWeights(conf.HIDDEN_LAYER_SIZE,
                                          conf.NUM_LABELS)
    init_W = np.hstack((init_W1.flatten(0), init_W2.flatten(0)))
    init_W = init_W.reshape((len(init_W), 1))

    print "Part 7: Implement Regularization\n"

    print "Checking Backpropagation\n"
    LEARN_RATE = 3
    from checkNNGradients import checkNNGradients
    checkNNGradients(LEARN_RATE)
    J, _ = NN.nnCostFunction(W, conf.INPUT_LAYER_SIZE, conf.HIDDEN_LAYER_SIZE,
                             conf.NUM_LABELS, X, y, LEARN_RATE)
    print ("Cost at parameters (loaded from w1.txt and w2.txt): %f"
           "\n(this value should be about 0.576051)\n") % J

    print "Part 8: Training NN\n"

    def costFunc(p):
        return NN.nnCostFunction(p, conf.INPUT_LAYER_SIZE,
                                 conf.HIDDEN_LAYER_SIZE, conf.NUM_LABELS,
                                 X, y, LEARN_RATE)

    LEARN_RATE = 1
    nn_params = NN.trainNN(costFunc, init_W, 400)
    W1 = np.reshape(nn_params[:conf.HIDDEN_LAYER_SIZE *
                              (conf.INPUT_LAYER_SIZE + 1)],
                    (conf.HIDDEN_LAYER_SIZE, (conf.INPUT_LAYER_SIZE + 1)))
    W2 = np.reshape(nn_params[conf.HIDDEN_LAYER_SIZE *
                              (conf.INPUT_LAYER_SIZE + 1):],
                    (conf.NUM_LABELS, (conf.HIDDEN_LAYER_SIZE + 1)))

    print "Part 9: Implement Predict\n"
    pred = NN.predict(W1, W2, X)
    print "Training Set Accuracy: %f\n" % ((pred == y).mean() * 100)
