import sofia
import numpy as np
import time
import sys

from sklearn.cross_validation import KFold
from sklearn.grid_search import IterGrid

if len(sys.argv) != 2:
    print '%s <svm-light-file>' % sys.argv[0]
    sys.exit(1)

data_file = sys.argv[1]

with open(data_file, 'r') as f:
    data = np.array(f.readlines())

print 'got %d data-points' % len(data)

# use default config (same as sofia-ml)
config = sofia.sofia_ml.SofiaConfig()

sofia.srand(31337)

param_errors = {}
cv = 5

param_grid = {
    'lambda' : [0.0001, 0.001, 0.01, 0.1],
}

for params in IterGrid(param_grid):
    cv_runs = []
    start = time.clock()

    for train, test in KFold(len(data), cv, indices=False):
        train_cv = sofia.SfDataSet(''.join(data[train]), True)
        test_cv = sofia.SfDataSet(''.join(data[test]), True)

        config.lambda_param = params['lambda']
        w = sofia.TrainModel(train_cv, config)
        error = sofia.sofia_ml.SvmObjective(test_cv, w, config)

        cv_runs.append(error)

    end = time.clock()

    mean_cv_error = sum(cv_runs)/len(cv_runs)
    print '\navg test error %.5f for %s' % (mean_cv_error, params)
    print '%d-fold CV took %f seconds' % (cv, (end-start))
    param_errors[mean_cv_error] = params

best_error = sorted(param_errors.keys())[0]
print '\nerror=%.5f for %s' % (best_error, param_errors[best_error])

