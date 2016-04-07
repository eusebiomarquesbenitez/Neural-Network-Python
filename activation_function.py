# coding: utf-8

import numpy as np


class Sigmoid(object):

    @staticmethod
    def y(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dy_dz(y):
        return y * (1. - y)
