#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration_Regression.h"

PFEnergyCalibration_Regression::PFEnergyCalibration_Regression(const std::string& modelFile, const std::string& inputOp, const std::string& outputOp) 
  : inputOp_(inputOp), outputOp_(outputOp), session_(nullptr)
{
  setUpTFlow(modelFile);
}

PFEnergyCalibration_Regression::~PFEnergyCalibration_Regression()
{
  if(session_)
    {
      TF_Status* status = TF_NewStatus();
      TF_CloseSession(session_, status);
      TF_DeleteSession(session_, status);
      TF_DeleteStatus(status);
    }
}


void 
PFEnergyCalibration_Regression::free_buffer(void* data, size_t length) {
  free(data);
}

TF_Buffer* 
PFEnergyCalibration_Regression::read_file(const std::string& file) {
  FILE* f = fopen(file.c_str(), "rb");

  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  //same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

void 
PFEnergyCalibration_Regression::setUpTFlow(const std::string& modelFile){  
  //Variable to hold tensorflow status
  TF_Status* status = TF_NewStatus();
        
  //get the grafdef from the file
  TF_Buffer* graph_def = read_file(modelFile);
        
  // Import graph_def into graph
  TF_Graph* graph = TF_NewGraph();
  TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
  TF_DeleteImportGraphDefOptions(graph_opts);
  TF_DeleteBuffer(graph_def);
        
  //Create tensorflow session from imported graph
  TF_SessionOptions* sess_opts = TF_NewSessionOptions();
  uint8_t config[] = {0x10, 0x01};
  TF_SetConfig(sess_opts, static_cast<void*>(config), 2, status);
  session_ = TF_NewSession(graph, sess_opts, status);
  TF_DeleteSessionOptions(sess_opts);
        
  TF_Operation* op_x = TF_GraphOperationByName(graph, inputOp_.c_str());
  TF_Operation* op_y = TF_GraphOperationByName(graph, outputOp_.c_str());
        
  //Clean up graph
  TF_DeleteGraph(graph);
        
  inputs_ .emplace_back(TF_Output({op_x, 0}));
  outputs_.emplace_back(TF_Output({op_y, 0}));
  targets_.emplace_back(op_y);
        
  TF_DeleteStatus(status);  
}  

const double 
PFEnergyCalibration_Regression::getRegression(std::vector<std::pair<std::string, double>> inputs) {
  //tensorflow status variable
  TF_Status* status = TF_NewStatus();
        
  //Create place to store the output vectors 
  std::vector<TF_Tensor*> output_values(1);

  //Construct tensorflow input tensor
  std::vector<TF_Tensor*> input_values;
  const int elemSize = sizeof(float);
  std::vector<int64_t> dims = {static_cast<int64_t>(1), static_cast<int64_t>(inputs.size())};
  int nelem = 1;
  for(const auto dimLen : dims) nelem *= dimLen;
  TF_Tensor* input_values_0 =  TF_AllocateTensor(TF_FLOAT, dims.data(), dims.size(), elemSize*nelem);
        
  input_values = { input_values_0 };
  float* basePtr = static_cast<float*>(TF_TensorData(input_values_0));
  for(unsigned int i=0; i < inputs.size(); i++)
  {
    *(basePtr+i) = inputs.at(i).second;
  }

  //predict values
  TF_SessionRun(session_,
		// RunOptions
		nullptr,
		// Input tensors
		inputs_.data(), input_values.data(), inputs_.size(),
		// Output tensors
		outputs_.data(), output_values.data(), outputs_.size(),
		// Target operations
		targets_.data(), targets_.size(),
		// RunMetadata
		nullptr,
		// Output status
		status);
        
  //Get output discriminators 
  auto discriminators = static_cast<float*>(TF_TensorData(output_values[0]));                
        
  //discriminators is a 2D array, we only want the first entry of every array
  double discriminator = static_cast<double>(discriminators[0]);

  for(auto* tensor : input_values)  TF_DeleteTensor(tensor);
  for(auto* tensor : output_values) TF_DeleteTensor(tensor);
        
  TF_DeleteStatus(status);

  return discriminator;
}

void 
PFEnergyCalibration_Regression::energyEmHadRegression(double t, double h, double e, double eta, double phi) {
  double tt = t;
  double hh = h;
  double ee = e;

  // Using TFlow to return a corrected total Energy
  std::vector<std::pair<std::string, double>> inputs = {
    {"p", tt},
    {"pf_totalRaw", ee+hh},
    {"pf_ecalRaw",    ee},
    {"pf_hcalRaw",    hh},
    {"eta",         eta},
    {"phi",         phi}
  };    
  double corrE = getRegression(inputs);
    
  // Post Process the tensorflow output
  corrE = 1/((corrE+1)/(ee+hh));

  // return Calibrated EcalE and HcalE
  e = ee/(ee+hh)*corrE;
  h = corrE - e;   
}
