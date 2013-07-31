"""
    Copyright 2013 Lyst Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


LOOP_BALANCED_STOCHASTIC = 1
LOOP_STOCHASTIC = 2

PREDICTION_LINEAR = 1
PREDICTION_LOGISTIC = 2

LEARNER_PEGASOS_SVM = 1
LEARNER_PEGASOS_LOGREG = 2

# defaults for hyperparameters
DFLT_ITERATIONS = 100000
DFLT_LAMBDA_REG = 0.1

# make sure we protect against lambda * eta > 1.0 which
# causes numerical issues for regularization and projection
MIN_SCALING_FACTOR = 0.0000001
MIN_SCALE = 0.00000000001

