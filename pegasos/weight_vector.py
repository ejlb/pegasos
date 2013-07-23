import numpy as np
import math

from . import constants

class WeightVector(object):
    def __init__(self, dimensionality):
        self._scale = 1.0
        self._squared_norm = 1.0
        self._weights = np.zeros(dimensionality)

    def _xi_squared_norm(self, xi):
        return np.sum(xi*xi)

    def _scale_to_one(self):
        self._weights *= self._scale
        self._scale = 1.0

    def scale(self, scaling_factor):
        if self._scale < constants.MIN_SCALE:
            self._scale_to_one()

        self._squared_norm *= math.pow(scaling_factor, 2)

        if scaling_factor > 0.0:
            self._scale *= scaling_factor
        else:
            raise ValueError('Scaling factor error, likely due \
                              to a large value eta * lambda')

    def add(self, xi, scaler):
        scaled_xi = xi * scaler
        inner = np.inner(self._weights, scaled_xi)
        self._weights += (scaled_xi / self._scale)

        self._squared_norm += self._xi_squared_norm(xi) \
                           * math.pow(scaler, 2) \
                           + (2.0 * self._scale * inner)

    def inner_product(self, x):
        return np.inner(self._weights, x)*self._scale

    def squared_norm(self):
        return self._squared_norm


