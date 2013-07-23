class PegasosSVMClassifier(SVMSofiaBase):
    def __init__(self,
                 iterations=100000,
                 dimensionality=2<<16,
                 lreg=0.1,
                 eta_type=ETA_PEGASOS,
                 loop_type=LOOP_BALANCED_STOCHASTIC):

        super(SVMSofiaBase, self).__init__(
                iterations,
                dimensionality,
                lreg,
                eta_type,
                sofia.sofia_ml.PEGASOS,
                loop_type)

class PegasosLogisticRegression(LogisticSofiaBase):
    def __init__(self,
                 iterations=100000,
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


