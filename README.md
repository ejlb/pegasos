sofiapy
=======

`sofiapy` is a set sklearn-like python binding for the google tool `sofia-ml`. These bindings do not include every feature of sofia-ml; the ranking and perceptron functionality are not included because the bindings were primary for the pegasos functionality. The bindings require a custom minimal version of sofia-ml (included), pybindgen and sklearn. See example/example.py for details.

* sofia.so is the raw bindings, can be imported and API matches the c++ API
* sofiapy is a wrapper round the raw bindings which have an sklearn-style interface

algorithm support
------------------
* learners: pegasos svm, logreg pegasos, lms pegasos, sgd svm, and logreg
* eta: basic, pegasos, constant
* loops: stochastic, balanced stochastic
* predictions: linear, logistic

API support
-----------
* load data
* train model
* save model
* load model
* predict
* srand bindings for seeding

config
------
Instead of command line arguments, sofiapy is controlled by a SofiaConfig struct. The structure
has the same defaults as the sofia-ml command line tool:

    iterations = 100000;
    dimensionality = 2<<16;
    lambda_param = 0.1;
    eta_type = PEGASOS_ETA;
    learner_type = PEGASOS;
    loop_type = STOCHASTIC;
    prediction_type = LINEAR;

The sklearn-like bindings do not need to use SofiaConfig directly.

todo
----
* predict_proba for SVM classifiers
