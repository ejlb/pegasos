from abc import ABCMeta, abstractmethod

from .base import PegasosBase
from . import constants
from . import pegasos


class SVMPegasosBase(PegasosBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 dimensionality,
                 lambda_reg,
                 eta_type,
                 learner_type,
                 loop_type):

        if learner_type != constants.PEGASOS_SVM:
            raise ValueError('%s only supports SVM learners' % self.__class__.__name__)

        super(SVMPegasosBase, self).__init__(
                iterations,
                dimensionality,
                lambda_reg,
                eta_type,
                learner_type,
                loop_type)

    def decision_function(self, X):
        if not self.weight_vector:
            raise ValueError('must call `fit` before `predict` or `decision_function`')

        return pegasos.predict_svm(self, X)

class LogisticPegasosBase(PegasosBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 dimensionality,
                 lambda_reg,
                 eta_type,
                 learner_type,
                 loop_type):

        if learner_type != constants.PEGASOS_LOGREG:
            raise ValueError('%s only supports logistic learners' % self.__class__.__name__)

        super(LogisticPegasosBase, self).__init__(
                iterations,
                dimensionality,
                lambda_reg,
                eta_type,
                learner_type,
                loop_type)

    def decision_function(self, X):
        if not self.weight_vector:
            raise ValueError('must call `fit` before `predict` or `decision_function`')

        # linear predictions

    def predict_proba(self, X):
        if not self.weight_vector:
            raise ValueError('must call `fit` before `predict_proba`')

        # probabalistic predictions

