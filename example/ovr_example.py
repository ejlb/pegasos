import sofia
import numpy as np
import time
import sys

from sklearn.cross_validation import KFold
from sklearn.grid_search import IterGrid

def _remap_labels(label_map, label):
    new_label_map = {}
    for k,v in label_map.items():
        if v == label:
            new_label_map[k] = 1.0
        else:
            new_label_map[k] = -1.0

    return new_label_map

def _remap_data(remapped_labels, data):
    #### use remapped_labels to build a new sf-dataset vi sf-spare-vector
    return data

def one_vs_all(train, test, config):
    train_sf = sofia.SfDataSet(''.join(train), True)
    test_sf = sofia.SfDataSet(''.join(test), True)

    label_map = dict([(i, train_sf.VectorAt(i).GetY())
        for i in range(train_sf.NumExamples())])
    label_set = set(label_map.values())

    w = []
    error = []

    ## re-map labels for 1-vs-all and average error
    for label in label_set:
        remapped_labels = _remap_labels(label_map, label)
        remapped_train = _remap_data(remapped_labels, train_sf)
        remapped_test = _remap_data(remapped_labels, test_sf)

        w.append(sofia.TrainModel(remapped_train, config))
        error.append(sofia.sofia_ml.SvmObjective(remapped_test, w[-1], config))

    return error, w

if len(sys.argv) != 2:
    print '%s <svm-light-file>' % sys.argv[0]
    sys.exit(1)

data_file = sys.argv[1]

with open(data_file, 'r') as f:
    data = np.array(f.readlines())

print 'got %d data-points' % len(data)

# use logistic pegasos with logistic predictions on a balanced minibatch training loop
config = sofia.sofia_ml.SofiaConfig()
config.learner_type = sofia.sofia_ml.LOGREG_PEGASOS;
config.loop_type = sofia.sofia_ml.BALANCED_STOCHASTIC;
config.prediction_type = sofia.sofia_ml.LOGISTIC;

sofia.srand(31337)

param_errors = {}
cv = 10

param_grid = {
    'lambda' : [0.00001, 0.0001, 0.001, 0.01, 0.1],
}

for params in IterGrid(param_grid):
    cv_runs = []
    start = time.clock()

    for train, test in KFold(len(data), cv, indices=False):
        config.lambda_param = params['lambda']

        errors, w = one_vs_all(data[train], data[test], config)
        error = sum(errors)/len(errors)
        cv_runs.append(error)

    end = time.clock()

    mean_cv_error = sum(cv_runs)/len(cv_runs)
    print 'avg test error %.5f for %s' % (mean_cv_error, params)
    print '%d-fold CV took %f seconds' % (cv, (end-start))
    param_errors[mean_cv_error] = params

best_error = sorted(param_errors.keys())[0]
print '\nerror=%.5f for %s' % (best_error, param_errors[best_error])

