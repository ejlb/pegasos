import time
import pegasos

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from scipy.sparse import csr_matrix

def fit():
    raw_data = load_digits(2)
    X = raw_data['data']
    y = raw_data['target']

    datasets = train_test_split(X, y, random_state=12345)
    sparse_datasets = map(csr_matrix, datasets)
    train_X, test_X, train_y, test_y = sparse_datasets
    test_y = test_y.todense()
    #train_X, test_X, train_y, test_y = datasets

    model = pegasos.PegasosLogisticRegression(iterations=2500)
    start = time.clock()
    #from profilehooks import profile
    #pf = profile(model.fit)
    #pf(train_X, train_y)
    model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    end = time.clock()

    print 'acc %.5f in %f seconds' % (score, end-start)

if __name__ == '__main__':
    fit()

