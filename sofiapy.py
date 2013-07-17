from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import sofia

### c++ sparse_vector interface
    # empty constructor
    # add(vector, label)
    # add(vector)

# prediction (how does it differ from objective, do we need our own objective?)

# probabalistic prediction for SVMs (override predict_proba)

# multi-label classification
# documentation
# tests
# configure script


class SofiaBase(BaseEstimator, ClassifierMixin):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=sofia.sofia_ml.PEGASOS_ETA,
                 learner_type=sofia.sofia_ml.PEGASOS,
                 loop_type=sofia.sofia_ml.STOCHASTIC):

        self.support_vectors = None
        self.sofia_config = sofia.sofia_ml.SofiaConfig()

        self.sofia_config.iterations = iterations
        self.sofia_config.dimensionality = dimensionality
        self.sofia_config.lambda_param = lreg
        self.sofia_config.eta_type = eta_type
        self.sofia_config.learner_type = learner_type
        self.sofia_config.loop_type = loop_type

    def _sofia_dataset(self, X, y=None):
        if np.all(y) and len(X) != len(y):
            raise ValueError('`X` and `y` must be the same length')

        sofia_dataset = sofia.SfDataSet(True)

        for i, xi in enumerate(X):
            yi = y[i] if np.all(y) != None else 0.0
            sparse_vector = sofia.SfSparseVector(xi, yi)
            sofia_dataset.AddLabeledVector(sparse_vector, yi)

        return sofia_dataset

    def fit(self, X, y):
        sofia_dataset = self._sofia_dataset(X, y)
        self.support_vectors = sofia.TrainModel(sofia_dataset, self.sofia_config)

    def predict(self, X):
        if not self.support_vectors:
            raise ValueError('must call `fit` before `predict`')

        sofia_X = self._sofia_dataset(X)
        self.sofia_config.prediction_type = sofia.sofia_ml.LINEAR
        return sofia.sofia.sofia_ml.SvmPredictionsOnTestSet(sofia_X, self.support_vectors, self.sofia_config)

    """
    def predict_proba(self, X):
        if not self.support_vectors:
            raise ValueError('must call `fit` before `predict_proba`')

        sofia_X = self._sofia_dataset(X)
        self.sofia_config.prediction_type = sofia.sofia_ml.LOGISTIC
        return sofia.sofia.sofia_ml.LogisticPredictionsOnTestSet(sofia_X, self.support_vectors, self.sofia_config)
    """


class PegasosSVMClassifier(SofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=sofia.sofia_ml.PEGASOS_ETA,
                 loop_type=sofia.sofia_ml.STOCHASTIC):

        super(PegasosSVMClassifier, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.PEGASOS,
                loop_type)


class PegasosLMSRegression(SofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=sofia.sofia_ml.PEGASOS_ETA,
                 loop_type=sofia.sofia_ml.STOCHASTIC):

        super(PegasosLMSRegression, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.LMS_REGRESSION,
                loop_type)


class PegasosLogisticRegression(SofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=sofia.sofia_ml.PEGASOS_ETA,
                 loop_type=sofia.sofia_ml.STOCHASTIC):

        super(PegasosLMSRegression, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.LOGREG_PEGASOS,
                loop_type)


class SGDSVMClassifier(SofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=sofia.sofia_ml.PEGASOS_ETA,
                 loop_type=sofia.sofia_ml.STOCHASTIC):

        super(SGDSVMClassifier, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.SDG_SVM,
                loop_type)


class ROMMAClassifier(SofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=sofia.sofia_ml.PEGASOS_ETA,
                 loop_type=sofia.sofia_ml.STOCHASTIC):

        super(ROMMAClassifier, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.ROMMA,
                loop_type)


class LogisticRegression(SofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=sofia.sofia_ml.PEGASOS_ETA,
                 loop_type=sofia.sofia_ml.STOCHASTIC):

        super(LogisticRegression, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.LOGREG,
                loop_type)

