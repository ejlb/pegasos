from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import atleast2d_or_csr

from . import pegasos, constants
from .weight_vector import WeightVector

import numpy as np

class PegasosBase(BaseEstimator, ClassifierMixin):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 dimensionality,
                 lreg,
                 eta_type,
                 learner_type,
                 loop_type):

        self.iterations = iterations
        self.dimensionality = dimensionality
        self.lreg = lreg
        self.eta_type = eta_type
        self.loop_type = loop_type
        self.learner_type = learner_type

        self.weight_vector = WeightVector(self.dimensionality)

    def fit(self, X, y):
        self._enc = LabelEncoder()
        y = self._enc.fit_transform(y)

        if len(self.classes_) != 2:
            raise ValueError("The number of classes must be 2, use sklearn.multiclass for more classes.")

        # the LabelEncoder maps the binary labels to 0 and 1 but the training
        # algorithm requires the labels to be -1 and +1
        y[y==0] = -1

        X = atleast2d_or_csr(X, dtype=np.float64, order="C")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n"
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))

        if self.loop_type == constants.LOOP_BALANCED_STOCHASTIC:
            pegasos.train_stochastic_balanced(self, X, y)
        elif self.loop_type == constants.LOOP_STOCHASTIC:
            pegasos.train_stochastic(self, X, y)
        else:
            raise ValueError('%s: unknown loop type' % self.loop_type)

        return self

    @abstractmethod
    def decision_function(self, X):
        raise NotImplemented

    def predict(self, X):
        return map(lambda x: 1 if x > 0 else 0, self.decision_function(X))

    @property
    def classes_(self):
        if not hasattr(self, '_enc'):
            raise ValueError('must call `fit` before `classes_`')
        return self._enc.classes_

