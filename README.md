pegasos
=======
`pegasos` is a python package for fitting SVM and logistic models via the pegasos solver. The package has an sklearn-like interface so can easily be used with existing sklearn functionality. The pegasos solver alternative between stochastic gradient descent and project steps. The number of training algorithm steps scales linearly with the regularization parameter lambda so the models are well suited to large datasets.

For details on the training algorithm see: http://eprints.pascal-network.org/archive/00004062/01/ShalevSiSr07.pdf. This implementation is based on the google tool `sofia-ml`

algorithm support
------------------
* learners: pegasos svm, pegasos logistic
* eta: basic, pegasos, constant
* loops: stochastic, balanced stochastic
* predictions: linear, logistic

see example.py for how to use the library

API support
-----------
* sparse or dense matrix support
* binary and multiclass model training
* balanced class weightings
* predictions (probabilistic predictions for logistic)
* model serialisation

speed
-----
On a 50000 x 25 matrix with 2 classes (via sklearn's `make\_classification`):

```
pegasos:   0.878 accuracy in   3.20 seconds
liblinear: 0.878 accuracy in   7.11 seconds
libsvm:    0.887 accuracy in 208.75 seconds
```

With 4 classes, training takes:

```
pegasos:   0.672 in  12.38 seconds
liblinear: 0.678 in  28.83 seconds
libsvm:    0.720 in 540.55 seconds
```

Pegasos and liblinear perform similarly in terms of accuracy but pegasos is much quicker and scales better. Libsvm is slighlty more accuracy but orders of magnitiude slower than both pegasos and liblinear.

todo
----
* predict\_proba for SVM classifiers

