import time
import sofiapy

from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import IterGrid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_classification

def fit():
    state=123

    X, y = make_classification(n_samples=50000, n_features=25, n_informative=20, n_classes=4, random_state=state)

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state)

    #model = OneVsRestClassifier(sofiapy.PegasosSVMClassifier())
    #model = LinearSVC()
    model = SVC(kernel='linear')

    start = time.clock()
    score = model.fit(train_X, train_y).score(test_X, test_y)
    end = time.clock()

    print '\ntest acc %.5f in %f seconds' % (score, end-start)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('fit()')
    fit()
