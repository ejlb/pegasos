from nose.tools import *
from sklearn.datasets import load_digits

import pegasos

def test_init():
    try:
        pegasos.PegasosSVMClassifier()
        pegasos.PegasosLogisticRegression()
    except:
        assert False, 'Failed to initialise classifiers'
