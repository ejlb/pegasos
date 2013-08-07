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


from .models import SVMPegasosBase, LogisticPegasosBase
from . import constants

class PegasosSVMClassifier(SVMPegasosBase):
    def __init__(self,
                 iterations=constants.DFLT_ITERATIONS,
                 lambda_reg=constants.DFLT_LAMBDA_REG,
                 loop_type=constants.LOOP_STOCHASTIC,
                 verbose=0):

        super(SVMPegasosBase, self).__init__(
                iterations,
                lambda_reg,
                constants.LEARNER_PEGASOS_SVM,
                loop_type,
                verbose)


class PegasosLogisticRegression(LogisticPegasosBase):
    def __init__(self,
                 iterations=constants.DFLT_ITERATIONS,
                 lambda_reg=constants.DFLT_LAMBDA_REG,
                 loop_type=constants.LOOP_STOCHASTIC,
                 verbose=0):

        super(PegasosLogisticRegression, self).__init__(
                iterations,
                lambda_reg,
                constants.LEARNER_PEGASOS_LOGREG,
                loop_type,
                verbose)

