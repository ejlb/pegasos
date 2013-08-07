import time
import pegasos

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

def fit():
    for samples in [1000, 10000, 100000, 1000000]:
        for sample_coef in [1,5,10,25]:
            models = {
                'sgd': SGDClassifier(power_t=1, learning_rate='invscaling', n_iter=sample_coef, eta0=0.01),
                'peg': pegasos.PegasosSVMClassifier(iterations=samples*sample_coef),
            }

            for k,v in models.items():
                X, y = make_classification(n_samples=samples, random_state=12345)
                datasets = train_test_split(X, y, random_state=12345)
                train_X, test_X, train_y, test_y = datasets

                start = time.clock()
                v.fit(train_X, train_y)
                end = time.clock()
                score = v.score(test_X, test_y)

                print '%s,%d,%d,%f,%f' % (k, samples, samples*sample_coef, score, end-start)

if __name__ == '__main__':
    fit()

