from abc import ABCMeta, abstractmethod

import numpy as np
import sofia

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import atleast2d_or_csr

# multi-label classification
    # use composition in base

# deal with sparsity (like svm-light -- sparse matrix)
# bias term/intercept
# load/save

# documentation
# tests
# configure script

ETA_BASIC = sofia.sofia_ml.BASIC_ETA
ETA_PEGASOS = sofia.sofia_ml.PEGASOS_ETA
ETA_CONSTANT = sofia.sofia_ml.CONSTANT

LOOP_BALANCED_STOCHASTIC = sofia.sofia_ml.BALANCED_STOCHASTIC
LOOP_STOCHASTIC = sofia.sofia_ml.STOCHASTIC

PREDICTION_LINEAR = sofia.sofia_ml.LINEAR
PREDICTION_LOGISTIC = sofia.sofia_ml.LOGISTIC


class SofiaBase(BaseEstimator, ClassifierMixin):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations=100000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=ETA_PEGASOS,
                 learner_type=sofia.sofia_ml.PEGASOS,
                 loop_type=LOOP_BALANCED_STOCHASTIC):

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
            # use dummy labels when predicting
            yi = y[i] if np.all(y) != None else 0.0

            ### need to handle this sparsly
            sparse_vector = sofia.SfSparseVector(list(xi), yi)
            sofia_dataset.AddLabeledVector(sparse_vector, yi)

        return sofia_dataset

    def fit(self, X, y):
        self._enc = LabelEncoder()
        y = self._enc.fit_transform(y)

        if len(self.classes_) < 2:
            raise ValueError("The number of classes has to be greater than"
                             " one.")

        X = atleast2d_or_csr(X, dtype=np.float64, order="C")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n"
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))

        sofia_dataset = self._sofia_dataset(X, y)

        self.support_vectors = sofia.TrainModel(sofia_dataset, self.sofia_config)
        return self

    def predict(self, X):
        if not self.support_vectors:
            raise ValueError('must call `fit` before `predict`')

        sofia_X = self._sofia_dataset(X)
        self.sofia_config.prediction_type = PREDICTION_LINEAR
        predictions = sofia.sofia_ml.SvmPredictionsOnTestSet(sofia_X, self.support_vectors)

        return map(lambda x: 1 if x > 0 else 0, list(predictions))

    @property
    def classes_(self):
        return self._enc.classes_

    def error(self, X, y):
        sofia_dataset = self._sofia_dataset(X, y)
        return sofia.sofia_ml.SvmObjective(sofia_dataset, self.support_vectors, self.sofia_config)


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

    def predict_proba(self, X):
        if not self.support_vectors:
            raise ValueError('must call `fit` before `predict`')

        sofia_X = self._sofia_dataset(X)
        self.sofia_config.prediction_type = PREDICTION_LOGISTIC
        return sofia.sofia_ml.SvmPredictionsOnTestSet(sofia_X, self.support_vectors)


class PegasosSVMClassifier(SVMSofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=ETA_PEGASOS,
                 loop_type=LOOP_BALANCED_STOCHASTIC):

        super(PegasosSVMClassifier, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.PEGASOS,
                loop_type)


class SGDSVMClassifier(SVMSofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=ETA_PEGASOS,
                 loop_type=LOOP_BALANCED_STOCHASTIC):

        super(SGDSVMClassifier, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.SGD_SVM,
                loop_type)


class PegasosLogisticRegression(LogisticSofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=ETA_PEGASOS,
                 loop_type=LOOP_BALANCED_STOCHASTIC):

        super(PegasosLogisticRegression, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.LOGREG_PEGASOS,
                loop_type)


class PegasosLMSRegression(LogisticSofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=ETA_PEGASOS,
                 loop_type=LOOP_BALANCED_STOCHASTIC):

        super(PegasosLMSRegression, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.LMS_REGRESSION,
                loop_type)


class LogisticRegression(LogisticSofiaBase):
    def __init__(self,
                 iterations=10000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=ETA_PEGASOS,
                 loop_type=LOOP_BALANCED_STOCHASTIC):

        super(LogisticRegression, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.LOGREG,
                loop_type)


