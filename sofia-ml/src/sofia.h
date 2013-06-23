#include "sofia-ml-methods.h"
#include "sf-weight-vector.h"

using std::string;

void SaveModelToFile(const string& file_name, SfWeightVector* w);
SfWeightVector *LoadModelFromFile(const string& file_name);
SfWeightVector *TrainModel (const SfDataSet& training_data, sofia_ml::SofiaConfig& config);
