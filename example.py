import time
import pegasos

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification

def fit():
    state=123

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, n_classes=2, random_state=state)

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state)

    model = pegasos.PegasosSVMClassifier(loop_type=pegasos.constants.LOOP_STOCHASTIC)

    start = time.clock()
    model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    end = time.clock()

    print '\ntest acc %.5f in %f seconds' % (score, end-start)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('fit()')
    fit()
