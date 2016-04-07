#!/usr/bin/env python
# coding:utf-8

import numpy as np
import conf
import NN
import utils


if __name__ == '__main__':
    # Part 1: Loading Data
    X, y = utils.loadData(conf.FILE_X, conf.FILE_Y)

    # Part 2: Loading Parameters
    W1, W2 = utils.loadParams(conf.FILE_W1, conf.FILE_W2)
    # Unroll parameters
    W = np.hstack((W1.flatten(0), W2.flatten(0)))
    W = W.reshape((len(W), 1))

    # Part 3: Compute Cost(Feedforward)
    LEARN_RATE = 0
    J, _ = NN.nnCostFunction(W, conf.INPUT_LAYER_SIZE, conf.HIDDEN_LAYER_SIZE,
                             conf.NUM_LABELS, X, y, LEARN_RATE)
    print "Cost at parameters (loaded from w1.txt and w2.txt): %f\
        \n(this value should be about 0.287629)\n" % J

    # Part 4: Implement Regularization
    LEARN_RATE = 1
    J, _ = NN.nnCostFunction(W, conf.INPUT_LAYER_SIZE, conf.HIDDEN_LAYER_SIZE,
                             conf.NUM_LABELS, X, y, LEARN_RATE)
    print 'Cost at parameters (loaded from w1.txt and w2.txt): %f\
        \n(this value should be about 0.383770)\n' % J

    # Part 6: Initializing Parameters
    init_W1 = utils.randInitializeWeights(conf.INPUT_LAYER_SIZE,
                                          conf.HIDDEN_LAYER_SIZE)
    init_W2 = utils.randInitializeWeights(conf.INPUT_LAYER_SIZE,
                                          conf.NUM_LABELS)

    # Part 7: Implement Regularization
    LEARN_RATE = 3
    from checkNNGradients import checkNNGradients
    checkNNGradients(LEARN_RATE)
    J, _ = NN.nnCostFunction(W, conf.INPUT_LAYER_SIZE, conf.HIDDEN_LAYER_SIZE,
                             conf.NUM_LABELS, X, y, LEARN_RATE)
    print 'Cost at parameters (loaded from w1.txt and w2.txt): %f\
        \n(this value should be about 0.576051)\n' % J
