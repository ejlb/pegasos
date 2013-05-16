import sofia
import numpy as np
import time
import sys

from sklearn.cross_validation import KFold
from sklearn.grid_search import IterGrid

def _remap_labels(data, label):
    label_map = {}

    for i in range(0, data.NumExamples()):
        y = data.VectorAt(i).GetY()
        label_map[y] = 1.0 if label == y else -1.0

    return label_map

def _remap_data(label_map, data):
    remapped_data = sofia.SfDataSet(True)

    for i in range(0, data.NumExamples()):
        sparse_vector = data.VectorAt(i)
        remapped_data.AddLabeledVector(sparse_vector,
                                       label_map[sparse_vector.GetY()])
    return remapped_data

def one_vs_all(train, test, config):
    train_sf = sofia.SfDataSet(''.join(train), True)
    test_sf = sofia.SfDataSet(''.join(test), True)

    label_set = set([train_sf.VectorAt(i).GetY()
            for i in range(train_sf.NumExamples())])

    w = []
    error = []

    for label in label_set:
        label_map = _remap_labels(train_sf, label)
        remapped_train = _remap_data(label_map, train_sf)
        remapped_test = _remap_data(label_map, test_sf)

        w.append(sofia.TrainModel(remapped_train, config))
        error.append(sofia.sofia_ml.SvmObjective(remapped_test, w[-1], config))

        # don't need 1-v-all for the binary labels
        if len(label_set) == 2:
            break

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
config.prediction_type = sofia.sofia_ml.LINEAR;

sofia.srand(31337)

param_errors = {}
cv = 2

param_grid = {
    'lambda' : [0.0001, 0.001, 0.01, 0.1],
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

