from nose.tools import *

import pegasos
import numpy as np

from scipy import sparse

def test_init():
    try:
        pegasos.PegasosSVMClassifier()
        pegasos.PegasosLogisticRegression()
    except:
        assert False, 'Failed to initialise classifiers'


def test_svm():
    X = np.array([[1,1,1],[1,1,0],[1,0,0],[0,0,0], [0,1,1], [0,0,1]])
    y = np.array([1,1,1,0,0,0])

    svm = pegasos.PegasosSVMClassifier(iterations=1000)
    svm.fit(X, y)

    assert np.all(svm.predict(X) == y)


def test_svm_sparse():
    X = sparse.csr_matrix([[1,1,1],[1,1,0],[1,0,0],[0,0,0], [0,1,1], [0,0,1]])
    y = sparse.csr_matrix([1,1,1,0,0,0])

    svm = pegasos.PegasosSVMClassifier(iterations=1000)
    svm.fit(X, y)

    assert np.all(svm.predict(X) == y.todense())


def test_logreg_probability():
    X = np.array([[1,1,1],[1,1,0],[1,0,0],[0,0,0], [0,1,1], [0,0,1]])
    y = np.array([1,1,1,0,0,0])

    logreg = pegasos.PegasosLogisticRegression(iterations=5000)
    logreg.fit(X, y)

    p = logreg.predict_proba(X)

    assert p[0][0] > p[0][1]
    assert p[1][0] > p[1][1]
    assert p[4][0] < p[4][1]
    assert p[5][0] < p[5][1]


def test_weight_vector():
    assert True

