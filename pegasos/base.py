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
                 lambda_reg,
                 eta_type,
                 learner_type,
                 loop_type):

        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.eta_type = eta_type
        self.loop_type = loop_type
        self.learner_type = learner_type

        self.weight_vector = None

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

        self.weight_vector = WeightVector(X.shape[1])

        if self.loop_type == constants.LOOP_BALANCED_STOCHASTIC:
            pegasos.train_stochastic_balanced(self, X, y)
        elif self.loop_type == constants.LOOP_STOCHASTIC:
            pegasos.train_stochastic(self, X, y)
        else:
            raise ValueError('%s: unknown loop type' % self.loop_type)

        return self

    def decision_function(self, X):
        if not self.weight_vector:
            raise ValueError('must call `fit` before `predict` or `decision_function`')

        return pegasos.predict(self, X)

    def predict(self, X):
        if not hasattr(self, '_enc'):
            raise ValueError('must call `fit` before `predict`')

        d = self.decision_function(X)
        d[d>0] = 1
        d[d<=0] = 0
        d = d.astype(np.int32, copy=False)

        return self._enc.inverse_transform(d)

    @property
    def classes_(self):
        if not hasattr(self, '_enc'):
            raise ValueError('must call `fit` before `classes_`')
        return self._enc.classes_

