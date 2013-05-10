#include "sf-hash-weight-vector.h"
#include "sofia-ml-methods.h"
#include "sf-weight-vector.h"

using std::string;

void SaveModelToFile(const string& file_name, SfWeightVector* w);
SfWeightVector *LoadModelFromFile(const string& file_name);
SfWeightVector *TrainModel (const SfDataSet& training_data, float lambda);
