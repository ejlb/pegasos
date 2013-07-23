from abc import ABCMeta, abstractmethod

import numpy as np
import sofia

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import atleast2d_or_csr

# speed
    # test again SVC/LinearSVC/samples/features

# deal with sparsity (like svm-light -- sparse matrix)
# predict with custom support-vectors (not fit/load from disk)
# bias term/intercept
# load/save

# documentation
# tests
# configure script



class SVMSofiaBase(SofiaBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 dimensionality,
                 lreg,
                 eta_type,
                 learner_type,
                 loop_type):

        if learner_type not in [sofia.sofia_ml.PEGASOS, sofia.sofia_ml.SGD_SVM]:
            raise ValueError('%s only supports SVM learners' % self.__class__.__name__)

        super(SVMSofiaBase, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                learner_type,
                loop_type)

    def predict(self, X):
        return map(lambda x: 1 if x > 0 else 0, self.decision_function(X))

    def decision_function(self, X):
        if not self.support_vectors:
            raise ValueError('must call `fit` before `predict` or `decision_function`')

        self.sofia_config.prediction_type = PREDICTION_LINEAR
        sofia_X = self._sofia_dataset(X)
        return np.array(list(sofia.sofia_ml.SvmPredictionsOnTestSet(sofia_X, self.support_vectors)))


class LogisticSofiaBase(SofiaBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 dimensionality,
                 lreg,
                 eta_type,
                 learner_type,
                 loop_type):

        if learner_type not in [
                sofia.sofia_ml.LOGREG,
                sofia.sofia_ml.LOGREG_PEGASOS,
                sofia.sofia_ml.LMS_REGRESSION]:
            raise ValueError('%s only supports logistic learners' % self.__class__.__name__)

        super(LogisticSofiaBase, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                learner_type,
                loop_type)

    def predict(self, X):
        return map(lambda x: 1 if x > 0 else 0, list(self.decision_function(X)))

    def decision_function(self, X):
        if not self.support_vectors:
            raise ValueError('must call `fit` before `predict` or `decision_function`')

        self.sofia_config.prediction_type = PREDICTION_LINEAR
        sofia_X = self._sofia_dataset(X)
        return sofia.sofia_ml.LogisticPredictionsOnTestSet(sofia_X, self.support_vectors)

    def predict_proba(self, X):
        if not self.support_vectors:
            raise ValueError('must call `fit` before `predict_proba`')

        self.sofia_config.prediction_type = PREDICTION_LOGISTIC
        sofia_X = self._sofia_dataset(X)
        return sofia.sofia_ml.LogisticPredictionsOnTestSet(sofia_X, self.support_vectors)


