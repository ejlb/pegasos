## speed tests
## sparsity
## train
## predict
## save/load

## docs
## setup.py/egg etc.

from .pegasos import SVMPegasosBase, LogisticPegasosBase
from . import constants

class PegasosSVMClassifier(SVMPegasosBase):
    def __init__(self,
                 iterations=constants.DFLT_ITERATIONS,
                 dimensionality=constants.DFLT_DIMENSIONALITY,
                 lambda_reg=constants.DFLT_LAMBDA_REG,
                 eta_type=constants.ETA_PEGASOS,
                 loop_type=constants.LOOP_BALANCED_STOCHASTIC):

        super(SVMPegasosBase, self).__init__(
                iterations,
                dimensionality,
                lambda_reg,
                eta_type,
                constants.LEARNER_PEGASOS_SVM,
                loop_type)


class PegasosLogisticRegression(LogisticPegasosBase):
    def __init__(self,
                 iterations=constants.DFLT_ITERATIONS,
                 dimensionality=constants.DFLT_DIMENSIONALITY,
                 lambda_reg=constants.DFLT_LAMBDA_REG,
                 eta_type=constants.ETA_PEGASOS,
                 loop_type=constants.LOOP_BALANCED_STOCHASTIC):

        super(PegasosLogisticRegression, self).__init__(
                iterations,
                dimensionality,
                lambda_reg,
                eta_type,
                constants.LEARNER_PERGASOS_LOGISTIC,
                loop_type)

