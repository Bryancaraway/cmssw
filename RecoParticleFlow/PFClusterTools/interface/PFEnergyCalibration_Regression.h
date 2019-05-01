#ifndef PFEnergyCalibration_Regression_h
#define PFEnergyCalibration_Regression_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"
#include "tensorflow/c/c_api.h"

#include "cstdlib"
#include "cstdio"
#include "cstring"
// Get Inputs that the network needs: totalRawE, HcalRawE, EcalRawE, eta, phi, p, Depth info
// These Inputs will be passed from and returned to PFAlgo

// Set up TensorFlow

// returns calibrated Hcal energy and calibrated Ecal energy
// Do this by CalibE = RawE/(RawE+RawH) * CorrTotalE
//            CalibH = CorrTotalE - CalibE


class PFEnergyCalibration_Regression
{
 public:

  PFEnergyCalibration_Regression(const std::string& modelFile = "/uscms_data/d3/bcaraway/MVA/PR_10_4_0_pre4/src/keras_frozen.pb", const std::string& inputOp = "main_input", const std::string& outputOp = "first_output/BiasAdd");
  ~PFEnergyCalibration_Regression();

  void energyEmHadRegression(double t, double h, double e, double eta, double phi);

 private:
  std::string inputOp_, outputOp_;

  //Tensoflow session pointer
  TF_Session* session_;

  std::vector<TF_Output>     inputs_;
  std::vector<TF_Output>     outputs_;
  std::vector<TF_Operation*> targets_;

  static void free_buffer(void* data, size_t length) ;

  TF_Buffer* read_file(const std::string& file);

  void setUpTFlow(const std::string& modelFile);

  const double getRegression(std::vector<std::pair<std::string, double>> inputs);

};
#endif
