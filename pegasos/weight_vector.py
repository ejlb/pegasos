import numpy as np
import math

from . import constants

class WeightVector(object):
    def __init__(self, dimensionality):
        self.scale = 1.0
        self.squared_norm = 1.0
        self.weights = np.zeros(dimensionality)

    def scale_to(self, scaling_factor):
        if self.scale < constants.MIN_SCALE:
            self.weights *= self.scale
            self.scale = 1.0

        self.squared_norm *= math.pow(scaling_factor, 2)

        if scaling_factor > 0.0:
            self.scale *= scaling_factor
        else:
            raise ValueError('Scaling factor error, likely due'
                             'to a large value eta * lambda')

    def add(self, xi, scaler):
        xi_scaled = xi * scaler

        inner = np.inner(self.weights, xi_scaled)
        self.weights += (xi_scaled / self.scale)

        self.squared_norm += np.sum(xi*xi) * math.pow(scaler, 2)
                          + (2.0 * self.scale * inner)

    def inner_product(self, x):
        return np.inner(self.weights, x)*self.scale

