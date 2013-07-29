import time
import pegasos

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

def fit():
    raw_data = load_digits(2)
    X = raw_data['data']
    y = raw_data['target']

    datasets = train_test_split(X, y, random_state=12345)
    train_X, test_X, train_y, test_y = datasets

    model = pegasos.PegasosLogisticRegression()
    start = time.clock()
    model.fit(train_X, train_y)
    end = time.clock()
    score = model.score(test_X, test_y)

    print 'acc %.5f in %f seconds' % (score, end-start)

if __name__ == '__main__':
    fit()

