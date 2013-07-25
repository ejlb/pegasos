from nose.tools import *
from sklearn.datasets import load_digits

import pegasos

def test_setup():
    pegasos.PegasosSVMClassifier()
    pegasos.PegasosLogisticRegression()
    return False
