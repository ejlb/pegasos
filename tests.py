import sofiapy
import unittest

from sklearn.datasets import load_digits

class TestSofiapy(unittest.TestCase):

    def setUp(self):
        self.models = [
            sofiapy.PegasosSVMClassifier,
            sofiapy.PegasosLMSRegression,
            sofiapy.PegasosLogisticRegression,
            sofiapy.SGDSVMClassifier,
            sofiapy.LogisticRegression,
        ]

        data = load_digits(2)
        self.X = data['data']
        self.y = data['target']

    def test_bias(self):
        for m in self.models:
            print m, m().fit(self.X, self.y).error(self.X, self.y)

if __name__ == '__main__':
    unittest.main()
