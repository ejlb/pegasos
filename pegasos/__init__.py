from .models import SVMPegasosBase, LogisticPegasosBase
from . import constants

class PegasosSVMClassifier(SVMPegasosBase):
    def __init__(self,
                 iterations=constants.DFLT_ITERATIONS,
                 lambda_reg=constants.DFLT_LAMBDA_REG,
                 loop_type=constants.LOOP_STOCHASTIC):

        super(SVMPegasosBase, self).__init__(
                iterations,
                lambda_reg,
                constants.LEARNER_PEGASOS_SVM,
                loop_type)


class PegasosLogisticRegression(LogisticPegasosBase):
    def __init__(self,
                 iterations=constants.DFLT_ITERATIONS,
                 lambda_reg=constants.DFLT_LAMBDA_REG,
                 loop_type=constants.LOOP_STOCHASTIC):

        super(PegasosLogisticRegression, self).__init__(
                iterations,
                lambda_reg,
                constants.LEARNER_PEGASOS_LOGREG,
                loop_type)

