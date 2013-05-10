sofiapy
=======

Partial python binding for sofia-ml. Currently the bindings are very basic and not every feature of 
sofia-ml is supported. The bindings require a custom minimal version of sofia-ml and pybindgen. See 
example/example.py for details

algorithm support
------------------
* learners: pegasos, logreg pegasos, logreg, lms, sgd svm, romma
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

