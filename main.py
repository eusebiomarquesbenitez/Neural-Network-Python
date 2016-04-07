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
    W = np.hstack((W1.flatten(1), W2.flatten(1)))
    W = W.reshape((len(W), 1))

    # Part 3: Compute Cost(Feedforward)
    J = NN.nnCostFunction(W1, W2,
                       conf.INPUT_LAYER_SIZE,
                       conf.HIDDEN_LAYER_SIZE,
                       conf.NUM_LABELS,
                       X, y,
                       conf.LEARN_RATE)
    print J
