import time
import sofiapy

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.cross_validation import KFold
from sklearn.grid_search import IterGrid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification

def fit():
    data = load_digits(4)
    X = data['data']
    y = data['target']

    X, y = make_classification(n_samples=10000, n_features=50, n_informative=40, n_classes=3)

    param_grid = {
        'lambda' : [0.0001, 0.001, 0.01, 0.1],
    }

    cv = 4
    param_errors = {}

    for params in IterGrid(param_grid):
        cv_runs = []
        start = time.clock()

        for train, test in KFold(len(X), cv, indices=False):
            train_X, train_y = X[train], y[train]
            test_X, test_y = X[test], y[test]
            model = OneVsRestClassifier(sofiapy.PegasosSVMClassifier(lreg=params['lambda']))
            #model = SVC()
            model.fit(train_X, train_y)
            cv_runs.append(model.score(test_X, test_y))

        end = time.clock()
        mean_cv_acc = sum(cv_runs)/len(cv_runs)
        print '\navg test acc %.5f for %s' % (mean_cv_acc, params)
        print '%d-fold CV took %f seconds' % (cv, (end-start))
        param_errors[mean_cv_acc] = params


    best_error = sorted(param_errors.keys(), reverse=True)[0]
    print '\nacc=%.5f for %s' % (best_error, param_errors[best_error])


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('fit()')
    fit()
