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
// sofia-ml.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Main file for stochastic active set svm (sofia-ml), 
// a variant of the PEGASOS stochastic gradient svm solver.

#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "sofia-ml-methods.h"
#include "sf-weight-vector.h"

using std::string;

void SaveModelToFile(const string& file_name, SfWeightVector* w) {
  std::fstream model_stream;
  model_stream.open(file_name.c_str(), std::fstream::out);
  if (!model_stream) {
    std::cerr << "Error opening model output file " << file_name << std::endl;
    return;
  }
  model_stream << w->AsString() << std::endl;
  model_stream.close();
}

SfWeightVector * LoadModelFromFile(const string& file_name) {
  std::fstream model_stream;
  model_stream.open(file_name.c_str(), std::fstream::in);
  if (!model_stream) {
    std::cerr << "Error opening model input file " << file_name << std::endl;
    return NULL;
  }

  string model_string;
  std::getline(model_stream, model_string);
  model_stream.close();

  SfWeightVector *w = new SfWeightVector(model_string);
  return w;
}

SfWeightVector *TrainModel (const SfDataSet& training_data, sofia_ml::SofiaConfig &config) {
  SfWeightVector* w = new SfWeightVector(config.dimensionality);

  if(config.loop_type == sofia_ml::STOCHASTIC) {
    sofia_ml::StochasticOuterLoop(training_data, 
                                  config.learner_type,
                                  config.eta_type,
                                  config.lambda_param,
                                  0,
                                  config.iterations,
                                  w);
  }
  else if(config.loop_type == sofia_ml::BALANCED_STOCHASTIC) {
    sofia_ml::BalancedStochasticOuterLoop(training_data, 
                                  config.learner_type,
                                  config.eta_type,
                                  config.lambda_param,
                                  0,
                                  config.iterations,
                                  w);
  }

  return w;
}
