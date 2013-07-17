import numpy as np
import time

import sofiapy

from sklearn.datasets import make_classification
from sklearn.cross_validation import KFold
from sklearn.grid_search import IterGrid

def _test_data():
    X, y = make_classification(
            n_features=2, n_redundant=0, n_informative=2,
            random_state=1, n_clusters_per_class=1)

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)

    return X, y

if __name__ == '__main__':
    X, y = _test_data()

    param_grid = {
        'lambda' : [0.0001, 0.001, 0.01, 0.1],
    }

    cv = 5
    param_errors = {}

    for params in IterGrid(param_grid):
        cv_runs = []
        start = time.clock()

        for train, test in KFold(len(X), cv, indices=False):
            train_X, train_y = X[train], y[train]
            test_X, test_y = X[test], y[test]

            model = sofiapy.PegasosSVMClassifier(lreg=params['lambda'])
            model.fit(train_X, train_y)
            error = model.score(test_X, test_y)
            cv_runs.append(error)

        end = time.clock()

        mean_cv_error = sum(cv_runs)/len(cv_runs)
        print '\navg test error %.5f for %s' % (mean_cv_error, params)
        print '%d-fold CV took %f seconds' % (cv, (end-start))
        param_errors[mean_cv_error] = params

    best_error = sorted(param_errors.keys())[0]
    print '\nerror=%.5f for %s' % (best_error, param_errors[best_error])
