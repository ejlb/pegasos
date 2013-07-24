""" Functions for fitting and predicting pegasos models. Heavily
    'inspired' by google's C++ sofia-ml implementation """

import random
import math
import numpy as np

from . import constants

def L2_regularize(w, eta, lambda_reg):
    scaling_factor = 1.0 - (eta * lambda_reg)
    w.scale_to(max(scaling_factor, constants.MIN_SCALING_FACTOR))

def etaval(eta_type, lambda_reg, iteration):
    """ return learning rate for current training iteration based on model.eta_type """
    if eta_type == constants.ETA_CONSTANT:
        return 0.02
    elif eta_type == constants.ETA_BASIC:
        return 10.0 / (iteration + 10.0)
    elif eta_type == constants.ETA_PEGASOS:
        return 1.0 / (lambda_reg * iteration)
    else:
        raise ValueError('%s: unknown eta type' % eta_type)

def pegasos_projection(w, lambda_reg):
    projection = 1.0 / math.sqrt(lambda_reg * w.squared_norm)
    if projection < 1.0:
        w.scale_to(projection)

def _single_svm_step(xi, yi, w, eta, lambda_reg):
    p = yi * w.inner_product(xi)
    L2_regularize(w, eta, lambda_reg)
    if p < 1.0 and yi != 0.0:
        w.add(xi, (eta * yi))
    pegasos_projection(w, lambda_reg)

def _single_logreg_step(xi, yi, w, eta, lambda_reg):
    loss = yi / (1 + np.exp(yi * w.inner_product(xi)))
    L2_regularize(w, eta, lambda_reg)
    w.add(xi, (eta * loss))
    pegasos_projection(w, lambda_reg)

def train_stochastic(model, X, y):
    for iteration in range(1, model.iterations):
        i = random.randint(0, X.shape[0]-1)

        xi = X[i]
        yi = y[i]

        eta = etaval(model.eta_type, model.lambda_reg, iteration)

        if model.learner_type == constants.LEARNER_PEGASOS_SVM:
            _single_svm_step(xi, yi, model.weight_vector, eta, model.lambda_reg)
        elif model.learner_type == constants.LEARNER_PEGASOS_LOGREG:
            _single_logreg_step(xi, yi, model.weight_vector, eta, model.lambda_reg)
        else:
            raise ValueError('%s: unknown learner type' % model.loop_type)

def train_stochastic_balanced(model, X, y):
    # get the indices of positive and negative cases
    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==-1)[0]

    for iteration in range(1, model.iterations):
        pos_i = np.random.choice(pos_idx)
        neg_i = np.random.choice(neg_idx)

        pos_xi = X[pos_i]
        pos_yi = y[pos_i]
        neg_xi = X[neg_i]
        neg_yi = y[neg_i]

        eta = etaval(model.eta_type, model.lambda_reg, iteration)

        if model.learner_type == constants.LEARNER_PEGASOS_SVM:
            _single_svm_step(pos_xi, pos_yi, model.weight_vector, eta, model.lambda_reg)
            _single_svm_step(neg_xi, neg_yi, model.weight_vector, eta, model.lambda_reg)
        elif model.learner_type == constants.LEARNER_PEGASOS_LOGREG:
            _single_logreg_step(pos_xi, neg_yi, model.weight_vector, eta, model.lambda_reg)
            _single_logreg_step(neg_xi, neg_yi, model.weight_vector, eta, model.lambda_reg)
        else:
            raise ValueError('%s: unknown learner type' % model.loop_type)

def predict(model, X):
    return np.dot(model.weight_vector.weights, X.T)

