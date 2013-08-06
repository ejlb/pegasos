pegasos
=======
pegasos is a pure-python package for fitting SVM and logistic models using the Primal Estimated sub-GrAdient SOlver. This implementation is based on the google tool `sofia-ml`. The package has an sklearn-like interface so can easily be used with existing sklearn functionality. At each training step, the pegasos solver randomly samples a batch from the training data. The runtime of the training algorithm scales linearly with the regularization parameter `lambda` and the number of training steps; as such the model is well suited to large datasets. For details on the training algorithm see: 

http://eprints.pascal-network.org/archive/00004062/01/ShalevSiSr07.pdf

API support
-----------
* sparse or dense matrix support
* binary classification (multiclass via sklearn.multiclass)
* balanced class weightings via training loops
* probabilistic predictions for logistic model
* model serialisation via cPickle

See `example.py` for how to use the library. 

speed
-----
There are benchmarks against sklearn's SGDClassifier in the benchmarks folder.

```
samples   pegasos  liblinear  libsvm
------------------------------------
10^4      4.08     0.55       10.42
10^5      4.09     17.35      2638.62
10^6      4.63     230.71     *
10^7      6.87     3318.32    *
```

\* `libsvm` times are missing because the models converge sometime around the heat-death of the universe

The near-constant training time of pegasos is due to the constant number of training steps. For larger datasets the number of iterations should be increased. A grid-search on the lambda regularization parameter may also be benifical. The accuracy of the classifiers is generally ordered as `libsvm` > `liblinear` > `pegasos` but the differences are only 0.5-1%

Note that training time will increase by a constant amount for sparse matrices

build
------
Requirements are:

* scikit-learn >= 0.13.1
* numpy >= 1.7.1
* scipy >= 0.10.1

and nose for tests:

```
python setup.py nosetests
```

todo
----
* more tests
* training batches (with online learning)

