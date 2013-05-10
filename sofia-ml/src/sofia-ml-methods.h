//===========================================================================//
// Copyright 2009 Google Inc.                                                //
//                                                                           //
// Licensed under the Apache License, Version 2.0 (the "License");           //
// you may not use this file except in compliance with the License.          //
// You may obtain a copy of the License at                                   //
//                                                                           //
//      http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                           //
// Unless required by applicable law or agreed to in writing, software       //
// distributed under the License is distributed on an "AS IS" BASIS,         //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
// See the License for the specific language governing permissions and       //
// limitations under the License.                                            //
//===========================================================================//
//
// sofia-ml-methods.h
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Non-member functions for training and applying classifiers in the
// sofia-ml suite.  This is the main API that callers will use.

#ifndef SOFIA_ML_METHODS_H__
#define SOFIA_ML_METHODS_H__

#include "sf-data-set.h"
#include "sf-sparse-vector.h"
#include "sf-weight-vector.h"

namespace sofia_ml {

  //--------------------------------------------------------------------------
  //                     Main API methods for Model Training
  //--------------------------------------------------------------------------
  // This section contains the Main API methods to call for training a model on
  // a given data set.
  //  For each method, the parameters are described as follows:
  //   training_set  an SfDataSet filled with labeled training data.
  //   learner_type  a LearnerType enum value (defined above) 
  //                   showing which learner to use.
  //   eta_type      an EtaType enum showing how to update the learning rate.
  //   lambda        regularization parameter (ignored by some LearnerTypes)
  //   c             capacity parameter (ignored by some LearnerTypes)
  //   num_iters     number of stochastic steps to take.

  // We currently support the following learners.
  enum LearnerType {
    PEGASOS,  // Pegasos SVM, using lambda as regularization parameter
    LOGREG_PEGASOS,  // Logistic Regression using Pegasos projection, and lambda as regularization parameter.
    LOGREG,  // Logistic Regression using lambda as regularization parameter.
    LMS_REGRESSION, // Least-mean-squares Regression (using Pegasos projection), and lambda
                    //as regularization parameter.
    SGD_SVM,  // Stochastic Gradient Descent SVM; lambda is regularization parameter.
    ROMMA  // ROMMA algorithm; takes no regularization parameter.
  };

  // Learning rate Eta may be set in different ways.
  enum EtaType {
    BASIC_ETA,  // On step i, eta = 1000 / (1000 + i)
    PEGASOS_ETA,  // On step i, eta = 1.0 / (lambda * i) 
    CONSTANT  // Use constant eta = 0.02 for all steps.
  };

  enum LoopType {
      STOCHASTIC,
      BALANCED_STOCHASTIC,
  };

  enum PredictionType {
      LINEAR,
      LOGISTIC,
  };

  struct SofiaConfig {
      unsigned int iterations;
      unsigned int dimensionality;
      float lambda_param;
      EtaType eta_type;
      LearnerType learner_type;
      LoopType loop_type;
      PredictionType prediction_type;

      SofiaConfig();
   };

  // Trains a model w over training_set, using learner_type and eta_type learner with
  // given parameters.  For each iteration, samples one example uniformly at random from
  // training set.  Each example in the training_set has an equal probability of being
  // selected on any given set; this is the training method to use for training a 
  // standard binary-class classifier.
  void StochasticOuterLoop(const SfDataSet& training_set,
                           LearnerType learner_type,
                           EtaType eta_type,
                           float lambda,
                           float c,
                           int num_iters,
                           SfWeightVector* w);

  // Trains a model w over training_set, using learner_type and eta_type learner with
  // given parameters.  For each iteration, samples one positive example uniformly at
  // random from the set of all positives, and samples one negative example uniformly
  // at random from the set of all negatives.  This can be useful for training binary-class
  // classifiers with a minority-class distribution.
  void BalancedStochasticOuterLoop(const SfDataSet& training_set,
                                   LearnerType learner_type,
                                   EtaType eta_type,
                                   float lambda,
                                   float c,
                                   int num_iters,
                                   SfWeightVector* w);

  //------------------------------------------------------------------------------//
  //                    Methods for Applying a Model on Data                      //
  //------------------------------------------------------------------------------//

  // Computes a single linear prediction, returning f(x) = < x, w >
  float SingleSvmPrediction(const SfSparseVector& x,
			    const SfWeightVector& w);

  // Computes a single linear prediction, returning f(x) = e(< x, w >) / (1.0 + e(< x, w >))
  float SingleLogisticPrediction(const SfSparseVector& x,
				 const SfWeightVector& w);

  // Performs a SingleSvmPrediction on each example in test_data.
  std::vector<float> SvmPredictionsOnTestSet(const SfDataSet& test_data, const SfWeightVector& w);

  // Performs a SingleLogisticPrediction on each example in test_data.
  std::vector<float> LogisticPredictionsOnTestSet(const SfDataSet& test_data, const SfWeightVector& w);

  // Computes the value of binary class SVM objective function on the given data set, given a
  // model w and a value of the regularization parameter lambda.
  float SvmObjective(const SfDataSet& data_set,
		     const SfWeightVector& w,
		     SofiaConfig &sofia);

  //--------------------------------------------------------------
  //          Single Stochastic Step Strategy Methods
  //--------------------------------------------------------------

  // Takes one step using the LearnerType defined by method, and returns true
  // iff the method took a gradient step (ie, modified the model).
  bool OneLearnerStep(LearnerType method,
		      const SfSparseVector& x,
		      float eta,
		      float c,
		      float lambda,
		      SfWeightVector* w);

  //------------------------------------------------------------------------------//
  //                         LearnerType Methods                                  //
  //------------------------------------------------------------------------------//

  // Takes a single PEGASOS step, including regularization and projection.
  // Returns true iff the example x was violating KKT conditions.
  bool SinglePegasosStep(const SfSparseVector& x,
			 float eta,
			 float lambda,
			 SfWeightVector* w);

  // Takes a single SGD SVM step, including regularization.
  // Returns true iff the example x was violating KKT conditions.
  bool SingleSgdSvmStep(const SfSparseVector& x,
			float eta,
			float lambda,
			SfWeightVector* w);

  // Takes a single PEGASOS step using logistic loss function (logistic
  // regression) rather than hinge loss function (SVM loss).  Includes
  // L2 regularization and projection.  Always returns true, as updates
  // are performed for all examples.
  bool SinglePegasosLogRegStep(const SfSparseVector& x,
			       float eta,
			       float lambda,
			       SfWeightVector* w);

  // Takes a single SDG step using logistic loss function (logistic
  // regression) rather than hinge loss function (SVM loss).  Includes
  // L2 regularization. Always returns true.
  bool SingleLogRegStep(const SfSparseVector& x,
			float eta,
			float lambda,
			SfWeightVector* w);
  
  // Takes a single PEGASOS step using least-mean-squares objective function
  // rather than hinge loss function (SVM loss).  Includes L2 regularization
  // and projection.  Always returns true, as updates are performed for all
  // examples.
  bool SingleLeastMeanSquaresStep(const SfSparseVector& x,
				  float eta,
				  float lambda,
				  SfWeightVector* w);

  // Takes a single ROMMA step.
  bool SingleRommaStep(const SfSparseVector& x,
		       SfWeightVector* w);

  //-------------------------------------------------------------------
  //                    Non-Member Utility Functions
  //-------------------------------------------------------------------

  // Performs the PEGASOS projection step, projecting w back so that it
  // in the feasible set of solutions.
  void PegasosProjection(float lambda,
			 SfWeightVector* w);

  // Perform L2 regularization step, penalizing squared norm of
  // weight vector w.  Note that this regularization step is accomplished
  // by using w <- (1 - (eta * lambda)) * w, but we use MIN_SCALING_FACTOR
  // if (1 - (eta * lambda)) < MIN_SCALING_FACTOR to prevent numerical
  // problems.
  void L2Regularize(float eta, float lambda, SfWeightVector* w);

  // Performs L2 regularization penalization for a total of effective_steps
  // implied steps.  Does so by using:
  // w <- w * ((1 - (eta * lambda)) ^ effective_steps)
  void L2RegularizeSeveralSteps(float eta,
				float lambda,
				float effective_steps,
				SfWeightVector* w);
  
}  // namespace sofia_ml

#endif  // SOFIA_ML_METHODS_H__
