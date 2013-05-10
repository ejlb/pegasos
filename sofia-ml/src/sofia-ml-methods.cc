//================================================================================//
// Copyright 2009 Google Inc.                                                     //
//                                                                                // 
// Licensed under the Apache License, Version 2.0 (the "License");                //
// you may not use this file except in compliance with the License.               //
// You may obtain a copy of the License at                                        //
//                                                                                //
//      http://www.apache.org/licenses/LICENSE-2.0                                //
//                                                                                //
// Unless required by applicable law or agreed to in writing, software            //
// distributed under the License is distributed on an "AS IS" BASIS,              //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.       //
// See the License for the specific language governing permissions and            //
// limitations under the License.                                                 //
//================================================================================//
//
// sofia-ml-methods.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Implementation of sofia-ml-methods.h

#include "sofia-ml-methods.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

// The MIN_SCALING_FACTOR is used to protect against combinations of
// lambda * eta > 1.0, which will cause numerical problems for regularization
// and PEGASOS projection.  
#define MIN_SCALING_FACTOR 0.0000001

namespace sofia_ml {
  
  // --------------------------------------------------- //
  //         Helper functions (Not exposed in API)
  // --------------------------------------------------- //

  int RandInt(int num_vals) {
    return static_cast<int>(rand()) % num_vals;
  }

  float RandFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
  }

  const SfSparseVector& RandomExample(const SfDataSet& data_set) {
    int num_examples = data_set.NumExamples();
    int i = static_cast<int>(rand()) % num_examples;
    if (i < 0) {
      i += num_examples;
    }
    return data_set.VectorAt(i);
  }

  inline float GetEta (EtaType eta_type, float lambda, int i) {
    switch (eta_type) {
    case BASIC_ETA:
      return 10.0 / (i + 10.0);
      break;
    case PEGASOS_ETA:
      return 1.0 / (lambda * i);
      break;
    case CONSTANT:
      return 0.02;
      break;
    default:
      std::cerr << "EtaType " << eta_type << " not supported." << std::endl;
      exit(0);
    }
    std::cerr << "Error in GetEta, we should never get here." << std::endl;
    return 0;
  }
  
  // --------------------------------------------------- //
  //            Stochastic Loop Strategy Functions
  // --------------------------------------------------- //

  void StochasticOuterLoop(const SfDataSet& training_set,
			   LearnerType learner_type,
			   EtaType eta_type,
			   float lambda,
			   float c,
			   int num_iters,
			   SfWeightVector* w) {
    for (int i = 1; i <= num_iters; ++i) {
      int random_example = static_cast<int>(rand()) % training_set.NumExamples();
      const SfSparseVector& x = training_set.VectorAt(random_example);
      float eta = GetEta(eta_type, lambda, i);
      OneLearnerStep(learner_type, x, eta, c, lambda, w);
    }
  }  

  void BalancedStochasticOuterLoop(const SfDataSet& training_set,
				   LearnerType learner_type,
				   EtaType eta_type,
				   float lambda,
				   float c,
				   int num_iters,
				   SfWeightVector* w) {
    // Create index of positives and negatives for fast sampling
    // of disagreeing pairs.
    vector<int> positives;
    vector<int> negatives;
    for (int i = 0; i < training_set.NumExamples(); ++i) {
      if (training_set.VectorAt(i).GetY() > 0.0)
	positives.push_back(i);
      else
	negatives.push_back(i);
    }

    // For each iteration, randomly sample one positive and one negative and
    // take one gradient step for each.
    for (int i = 1; i <= num_iters; ++i) {
      float eta = GetEta(eta_type, lambda, i);

      const SfSparseVector& pos_x =
	training_set.VectorAt(positives[RandInt(positives.size())]);
      OneLearnerStep(learner_type, pos_x, eta, c, lambda, w);

      const SfSparseVector& neg_x =
	training_set.VectorAt(negatives[RandInt(negatives.size())]);
      OneLearnerStep(learner_type, neg_x, eta, c, lambda, w);
    }
  }

  //------------------------------------------------------------------------------//
  //                    Methods for Applying a Model on Data                      //
  //------------------------------------------------------------------------------//

  float SingleSvmPrediction(const SfSparseVector& x,
			    const SfWeightVector& w) {
    return w.InnerProduct(x);
  }

  float SingleLogisticPrediction(const SfSparseVector& x,
				 const SfWeightVector& w) {
    float p = w.InnerProduct(x);
    return exp(p) / (1.0 + exp(p));
  }
  
  std::vector<float> SvmPredictionsOnTestSet(const SfDataSet& test_data, const SfWeightVector& w) {
    std::vector<float> predictions;
    int size = test_data.NumExamples();
    for (int i = 0; i < size; ++i) {
      predictions.push_back(w.InnerProduct(test_data.VectorAt(i)));
    }
    return predictions;
  }

  std::vector<float> LogisticPredictionsOnTestSet(const SfDataSet& test_data, const SfWeightVector& w) {
    vector<float> predictions;
    int size = test_data.NumExamples();
    for (int i = 0; i < size; ++i) {
      predictions.push_back(SingleLogisticPrediction(test_data.VectorAt(i), w));
    }
    return predictions;
  }

  float SvmObjective(const SfDataSet& data_set,
		     const SfWeightVector& w,
             SofiaConfig &config) {
    vector<float> predictions;
    predictions = SvmPredictionsOnTestSet(data_set, w);
    float objective = w.GetSquaredNorm() * config.lambda_param / 2.0;
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      float loss_i = 1.0 - (predictions[i] * data_set.VectorAt(i).GetY());
      float incremental_loss = (loss_i < 0.0) ? 
	0.0 : loss_i / data_set.NumExamples();
      objective += incremental_loss;
    }
    return objective;
  }

  // --------------------------------------------------- //
  //       Single Stochastic Step Strategy Methods
  // --------------------------------------------------- //
  
  bool OneLearnerStep(LearnerType learner_type,
		      const SfSparseVector& x,
		      float eta,
		      float c,
		      float lambda,
		      SfWeightVector* w) {
    switch (learner_type) {
    case PEGASOS:
      return SinglePegasosStep(x, eta, lambda, w);
    case LOGREG_PEGASOS:
      return SinglePegasosLogRegStep(x, eta, lambda, w);
    case LOGREG:
      return SingleLogRegStep(x, eta, lambda, w);
    case LMS_REGRESSION:
      return SingleLeastMeanSquaresStep(x, eta, lambda, w);
    case SGD_SVM:
      return SingleSgdSvmStep(x, eta, lambda, w);
    case ROMMA:
      return SingleRommaStep(x, w);
    default:
      std::cerr << "Error: learner_type " << learner_type
		<< " not supported." << std::endl;
      exit(0);
    }
  }

  // --------------------------------------------------- //
  //            Single Stochastic Step Functions
  // --------------------------------------------------- //
  
  bool SinglePegasosStep(const SfSparseVector& x,
			  float eta,
			  float lambda,
			  SfWeightVector* w) {
    float p = x.GetY() * w->InnerProduct(x);    

    L2Regularize(eta, lambda, w);    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.GetY() != 0.0) {
      w->AddVector(x, (eta * x.GetY())); 
    }

    PegasosProjection(lambda, w);
    return (p < 1.0 && x.GetY() != 0.0);
  }

  bool SingleRommaStep(const SfSparseVector& x,
		       SfWeightVector* w) {
    float wx = w->InnerProduct(x);
    float p = x.GetY() * wx;
    const float kVerySmallNumber = 0.0000000001;
    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.GetY() != 0.0) {
      float xx = x.GetSquaredNorm();
      float ww = w->GetSquaredNorm();
      float c = ((xx * ww) - p + kVerySmallNumber) /
	((xx * ww) - (wx * wx) + kVerySmallNumber);

      float d = (ww * (x.GetY() - wx) + kVerySmallNumber) /
	((xx * ww) - (wx * wx) + kVerySmallNumber);

      // Avoid numerical problems caused by examples of extremely low magnitude.
      if (c >= 0.0) {
	w->ScaleBy(c);
	w->AddVector(x, d); 
      }
    }

    return (p < 1.0 && x.GetY() != 0.0);
  }

  bool SingleSgdSvmStep(const SfSparseVector& x,
			  float eta,
			  float lambda,
			  SfWeightVector* w) {
    float p = x.GetY() * w->InnerProduct(x);    

    L2Regularize(eta, lambda, w);    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.GetY() != 0.0) {
      w->AddVector(x, (eta * x.GetY())); 
    }

    return (p < 1.0 && x.GetY() != 0.0);
  }

  bool SinglePegasosLogRegStep(const SfSparseVector& x,
			       float eta,
			       float lambda,
			       SfWeightVector* w) {
    float loss = x.GetY() / (1 + exp(x.GetY() * w->InnerProduct(x)));

    L2Regularize(eta, lambda, w);    
    w->AddVector(x, (eta * loss));
    PegasosProjection(lambda, w);
    return (true);
  }

  bool SingleLogRegStep(const SfSparseVector& x,
			float eta,
			float lambda,
			SfWeightVector* w) {
    float loss = x.GetY() / (1 + exp(x.GetY() * w->InnerProduct(x)));

    L2Regularize(eta, lambda, w);    
    w->AddVector(x, (eta * loss));
    return (true);
  }

  bool SingleLeastMeanSquaresStep(const SfSparseVector& x,
				  float eta,
				  float lambda,
				  SfWeightVector* w) {
    float loss = x.GetY() - w->InnerProduct(x);
    L2Regularize(eta, lambda, w);    
    w->AddVector(x, (eta * loss));
    PegasosProjection(lambda, w);
    return (true);
  }

  void L2Regularize(float eta, float lambda, SfWeightVector* w) {
    float scaling_factor = 1.0 - (eta * lambda);
    if (scaling_factor > MIN_SCALING_FACTOR) {
      w->ScaleBy(1.0 - (eta * lambda));  
    } else {
      w->ScaleBy(MIN_SCALING_FACTOR); 
    }
  }

  void L2RegularizeSeveralSteps(float eta,
				float lambda,
				float effective_steps,
				SfWeightVector* w) {
    float scaling_factor = 1.0 - (eta * lambda);
    scaling_factor = pow(scaling_factor, effective_steps);
    if (scaling_factor > MIN_SCALING_FACTOR) {
      w->ScaleBy(1.0 - (eta * lambda));  
    } else {
      w->ScaleBy(MIN_SCALING_FACTOR); 
    }
  }
  
  void PegasosProjection(float lambda, SfWeightVector* w) {
    float projection_val = 1 / sqrt(lambda * w->GetSquaredNorm());
    if (projection_val < 1.0) {
      w->ScaleBy(projection_val);
    }
  }
  
  SofiaConfig::SofiaConfig() {
      iterations = 100000;
      dimensionality = 2<<16;
      lambda_param = 0.1;
      eta_type = PEGASOS_ETA;
      learner_type = PEGASOS;
      loop_type = STOCHASTIC;
      prediction_type = LINEAR;
   }

}  // namespace sofia_ml
  
