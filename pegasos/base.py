from . import constants

class PegasosBase(BaseEstimator, ClassifierMixin):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self,
                 iterations,
                 dimensionality,
                 lreg,
                 eta_type,
                 learner_type,
                 loop_type):

        self.support_vectors = None
        self.sofia_config = sofia.sofia_ml.SofiaConfig()

        self.iterations = self.sofia_config.iterations = iterations
        self.dimensionality = self.sofia_config.dimensionality = dimensionality
        self.lreg = self.sofia_config.lambda_param = lreg
        self.eta_type = self.sofia_config.eta_type = eta_type
        self.loop_type = self.sofia_config.loop_type = loop_type
        self.sofia_config.learner_type = learner_type

    def _sofia_dataset(self, X, y=None):
        if np.all(y) and len(X) != len(y):
            raise ValueError('`X` and `y` must be the same length')

        sofia_dataset = sofia.SfDataSet(True)

        for i, xi in enumerate(X):
            yi = y[i] if np.all(y) else 0.0
            sparse_vector = sofia.SfSparseVector(list(xi), yi)
            sofia_dataset.AddLabeledVector(sparse_vector, yi)

        return sofia_dataset

    def fit(self, X, y):
        self._enc = LabelEncoder()
        y = self._enc.fit_transform(y)

        if len(self.classes_) != 2:
            raise ValueError("The number of classes must be 2, use sklearn.multiclass for more classes.")

        """
        the LabelEncoder maps the binary labels to 0 and 1 but the svmlight
        format used by sofia-ml requires the labels to be -1 and +1
        """
        y[y==0] = -1

        X = atleast2d_or_csr(X, dtype=np.float64, order="C")

