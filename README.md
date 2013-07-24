pegasos
=======
`pegasos` is a python package for fitting SVM and logistic models via the pegasos solver. The package has an sklearn-like interface so can easily be used with existing sklearn functionality. The pegasos solver alternative between stochastic gradient descent and project steps. The number of training algorithm steps scales linearly with the regularization parameter lambda and the number of iterations; as such the model is well suited to large datasets.

For details on the training algorithm see: 

http://eprints.pascal-network.org/archive/00004062/01/ShalevSiSr07.pdf. 

This implementation is based on the google tool `sofia-ml`

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
```
samples   pegasos  liblinear  libsvm
------------------------------------
10^4      4.08     0.55       10.42
10^5      4.09     17.35      2638.62
10^6      4.63     230.71     *
10^7      6.87     3318.32    *
```

\* libsvm times are missing because the models converge sometime around the heat-death of the universe

The constant training time of pegasos is due to keeping a constant number of iterations. For larger datasets the number of iterations should be increased. A grid-search on the lambda regularization parameter may also be benifical. The accuracy of the classifiers is generally `libsvm` > `liblinear` > `pegasos` but the differences are only 0.5-1%

requirements
------------
* scikit-learn >= 0.13.1

todo
----
* predict\_proba for SVM classifiers

