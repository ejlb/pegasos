import time
import pegasos

from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits, make_classification
from sklearn.linear_model import SGDClassifier

def fit():
    """
    raw_data = load_digits(2)
    X = raw_data['data']
    y = raw_data['target']
    """

    for samples in [1000, 10000, 100000, 1000000, 10000000]:
        for iterations in [10, 100, 1000, 10000]:

            models = {
                'sgd': SGDClassifier(power_t=1, learning_rate='invscaling', n_iter=iterations, eta0=0.01),
                'peg': pegasos.PegasosSVMClassifier(iterations=iterations),
            }

            for k,v in models.items():
                X, y = make_classification(n_samples=samples, random_state=12345)
                datasets = train_test_split(X, y, random_state=12345)
                train_X, test_X, train_y, test_y = datasets

                start = time.clock()
                v.fit(train_X, train_y)
                end = time.clock()
                score = v.score(test_X, test_y)

                print '%s,%d,%d,%f,%f' % (k, samples, iterations, score, end-start)

if __name__ == '__main__':
    fit()

