import numpy as np
import math

from scipy import sparse

from . import constants

class WeightVector(object):
    def __init__(self, X):
        self.scale = 1.0
        self.squared_norm = 1.0
        self.dimensionality = X.shape[1]

        if sparse.issparse(X):
            self.weights = sparse.csr_matrix(np.zeros(self.dimensionality))
        else:
            self.weights = np.zeros(self.dimensionality)

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
        if sparse.issparse(xi):
            xi_scaled = sparse.csr_matrix(xi.T * scaler)
            inner = (self.weights * xi_scaled).sum()
        else:
            xi_scaled = xi * scaler
            inner = np.inner(self.weights, xi_scaled)

        self.weights = self.weights + (xi_scaled / self.scale).T

        xi_inner = (xi*xi.T).sum()
        self.squared_norm += xi_inner \
                          *  math.pow(scaler, 2) \
                          +  (2.0 * self.scale * inner)

    def inner_product(self, x):
        if sparse.issparse(x):
            return self.weights*x.T*self.scale
        else:
            return np.inner(self.weights, x)*self.scale

