from abc import ABCMeta, abstractmethod

from .base import PegasosBase
from . import constants

import numpy as np

class SVMPegasosBase(PegasosBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 lambda_reg,
                 learner_type,
                 loop_type):

        if learner_type != constants.LEARNER_PEGASOS_SVM:
            raise ValueError('%s only supports SVM learners' % self.__class__.__name__)

        super(SVMPegasosBase, self).__init__(
                iterations,
                lambda_reg,
                learner_type,
                loop_type)


class LogisticPegasosBase(PegasosBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 lambda_reg,
                 learner_type,
                 loop_type):

        if learner_type != constants.LEARNER_PEGASOS_LOGREG:
            raise ValueError('%s only supports logistic learners' % self.__class__.__name__)

        super(LogisticPegasosBase, self).__init__(
                iterations,
                lambda_reg,
                learner_type,
                loop_type)

    def predict_proba(self, X):
        if not self.weight_vector:
            raise ValueError('must call `fit` before `predict_proba`')

        p = self.decision_function(X) * self.weight_vector.scale
        positive = np.exp(p) / (1.0 + np.exp(p))

        # Return positive and negative class probabilities to
        # satisfy sklearn pipelines such as OneVsRestClassifier.
        return np.vstack((positive, 1 - positive)).T

