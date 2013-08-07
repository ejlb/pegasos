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
                 loop_type,
                 verbose):

        if learner_type != constants.LEARNER_PEGASOS_SVM:
            raise ValueError('%s only supports SVM learners' % self.__class__.__name__)

        super(SVMPegasosBase, self).__init__(
                iterations,
                lambda_reg,
                learner_type,
                loop_type,
                verbose)


class LogisticPegasosBase(PegasosBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 lambda_reg,
                 learner_type,
                 loop_type,
                 verbose):

        if learner_type != constants.LEARNER_PEGASOS_LOGREG:
            raise ValueError('%s only supports logistic learners' % self.__class__.__name__)

        super(LogisticPegasosBase, self).__init__(
                iterations,
                lambda_reg,
                learner_type,
                loop_type,
                verbose)

    def predict_proba(self, X):
        if not self.weight_vector:
            raise ValueError('must call `fit` before `predict_proba`')

        p = self.decision_function(X) * self.weight_vector.scale
        positive = np.exp(p) / (1.0 + np.exp(p))

        # Return positive and negative class probabilities to
        # satisfy sklearn pipelines such as OneVsRestClassifier.
        return np.vstack((positive, 1 - positive)).T

