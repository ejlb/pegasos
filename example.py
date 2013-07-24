import time
import pegasos

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def fit():
    state=12345

    for samples in [1000, 10000, 100000, 1000000]:
        models = {
            'pegasos-svm': OneVsRestClassifier(pegasos.PegasosSVMClassifier()),
            'pegasos-log': OneVsRestClassifier(pegasos.PegasosLogisticRegression()),
            'liblinear': LinearSVC(),
        }

        print '\n%d samples' % samples

        for k,v in models.items():
            X, y = make_classification(n_samples=samples, n_informative=15)
            train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state)

            start = time.clock()

            v.fit(train_X, train_y)
            score = v.score(test_X, test_y)

            end = time.clock()
            print '%s: acc %.5f in %f seconds' % (k, score, end-start)

if __name__ == '__main__':
    fit()

