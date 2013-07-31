"""
    Copyright 2013 Lyst Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


import numpy as np
import math

from scipy import sparse

from . import constants
from . import utils

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
        xi_scaled = xi * scaler
        self.weights = self.weights + (xi_scaled / self.scale)
        inner = utils.inner(self.weights, xi_scaled)

        self.squared_norm += utils.inner(xi, xi) \
                          *  math.pow(scaler, 2) \
                          +  (2.0 * self.scale * inner)

    def inner_product(self, x):
        return utils.inner(self.weights, x)*self.scale

