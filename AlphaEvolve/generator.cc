#include "generator.h"

#include "definitions.h"
#include "instruction.pb.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/memory/memory.h"
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <regex>
#include <sstream>
#include <stdio.h>

namespace alphaevolve {

using ::absl::make_unique;
using ::std::endl;
using ::std::make_shared;
using ::std::mt19937;
using ::std::shared_ptr;
using ::std::vector;
using ::std::string;

void PadComponentFunctionWithInstruction(
    const size_t total_instructions,
    const shared_ptr<const Instruction>& instruction,
    vector<shared_ptr<const Instruction>>* component_function) {
  component_function->reserve(total_instructions);
  while (component_function->size() < total_instructions) {
    component_function->emplace_back(instruction);
  }
}

Generator::Generator(
    const HardcodedAlgorithmID init_model,
    const IntegerT setup_size_init,
    const IntegerT predict_size_init,
    const IntegerT learn_size_init,
    const vector<Op>& allowed_setup_ops,
    const vector<Op>& allowed_predict_ops,
    const vector<Op>& allowed_learn_ops,
    mt19937* bit_gen,
    RandomGenerator* rand_gen,
    vector<std::string>* read_from_file)
    : init_model_(init_model),
      setup_size_init_(setup_size_init),
      predict_size_init_(predict_size_init),
      learn_size_init_(learn_size_init),
      allowed_setup_ops_(allowed_setup_ops),
      allowed_predict_ops_(allowed_predict_ops),
      allowed_learn_ops_(allowed_learn_ops),
      rand_gen_(rand_gen),
      randomizer_(
          allowed_setup_ops,
          allowed_predict_ops,
          allowed_learn_ops,
          bit_gen,
          rand_gen_),
      no_op_instruction_(make_shared<const Instruction>()),
      read_from_file_(*read_from_file) {}

Algorithm Generator::TheInitModel() {
  return ModelByID(init_model_);
}

Algorithm Generator::ModelByID(const HardcodedAlgorithmID model) {
  switch (model) {
    case NO_OP_ALGORITHM:
      return NoOp();
    case RANDOM_ALGORITHM:
      return Random();
    case MY_ALGORITHM:
      return My_Algorithm();
    case NEURAL_NET_ALGORITHM:
      return NeuralNet(
          kDefaultLearningRate, 0.1, 0.1);
    case NEURAL_NET_ALGORITHM2:
      return NeuralNet2(
          kDefaultLearningRate, 0.1, 0.1);
    case NEURAL_NET_ALGORITHM3:
      return NeuralNet3(
          kDefaultLearningRate, 0.1, 0.1);
    case MY_ALPHA:
      return MyNeuralNet();
    case ALPHA_26:
      return MyNeuralNet();
    case ALPHA_15:
      return MyNeuralNet();
    case ALPHA_13:
      return MyNeuralNet();
    case ALPHA_101:
      return MyNeuralNet();
    case ALPHA_38:
      return MyNeuralNet();
    case INTEGRATION_TEST_DAMAGED_NEURAL_NET_ALGORITHM: {
      Algorithm algorithm = NeuralNet(
          kDefaultLearningRate, 0.1, 0.1);
      // Delete the first two instructions in setup which are the
      // gaussian initialization of the first and final layer weights.
      algorithm.setup_.erase(algorithm.setup_.begin());
      algorithm.setup_.erase(algorithm.setup_.begin());
      return algorithm;
    }
    case LINEAR_ALGORITHM:
      return LinearModel(kDefaultLearningRate);
    default:
      LOG(FATAL) << "Unsupported algorithm ID." << endl;
  }
}

inline void FillComponentFunctionWithInstruction(
    const IntegerT num_instructions,
    const shared_ptr<const Instruction>& instruction,
    vector<shared_ptr<const Instruction>>* component_function) {
  component_function->reserve(num_instructions);
  component_function->clear();
  for (IntegerT pos = 0; pos < num_instructions; ++pos) {
    component_function->emplace_back(instruction);
  }
}

Algorithm Generator::NoOp() {
  Algorithm algorithm;
  FillComponentFunctionWithInstruction(
      setup_size_init_, no_op_instruction_, &algorithm.setup_);
  FillComponentFunctionWithInstruction(
      predict_size_init_, no_op_instruction_, &algorithm.predict_);
  FillComponentFunctionWithInstruction(
      learn_size_init_, no_op_instruction_, &algorithm.learn_);
  return algorithm;
}

Algorithm Generator::Random() {
  Algorithm algorithm = NoOp();
  CHECK(setup_size_init_ == 0 || !allowed_setup_ops_.empty());
  CHECK(predict_size_init_ == 0 || !allowed_predict_ops_.empty());
  CHECK(learn_size_init_ == 0 || !allowed_learn_ops_.empty());
  randomizer_.Randomize(&algorithm);
  return algorithm;
}

void PadComponentFunctionWithRandomInstruction(
    const size_t total_instructions, const Op op,
    RandomGenerator* rand_gen,
    vector<shared_ptr<const Instruction>>* component_function) {
  component_function->reserve(total_instructions);
  while (component_function->size() < total_instructions) {
    component_function->push_back(make_shared<const Instruction>(op, rand_gen));
  }
}

Generator::Generator()
    : init_model_(RANDOM_ALGORITHM),
      setup_size_init_(6),
      predict_size_init_(3),
      learn_size_init_(9),
      allowed_setup_ops_(
          {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
      allowed_predict_ops_(
          {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
      allowed_learn_ops_(
          {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
      bit_gen_owned_(make_unique<mt19937>(GenerateRandomSeed())),
      rand_gen_owned_(make_unique<RandomGenerator>(bit_gen_owned_.get())),
      rand_gen_(rand_gen_owned_.get()),
      randomizer_(
          allowed_setup_ops_,
          allowed_predict_ops_,
          allowed_learn_ops_,
          bit_gen_owned_.get(),
          rand_gen_),
      no_op_instruction_(make_shared<const Instruction>()) {}

Algorithm Generator::UnitTestNeuralNetNoBiasNoGradient(
    const double learning_rate) {
  Algorithm algorithm;

  // Scalar addresses
  constexpr AddressT kLearningRateAddress = 2;
  constexpr AddressT kPredictionErrorAddress = 3;
  CHECK_GE(kMaxScalarAddresses, 4);

  // Vector addresses.
  constexpr AddressT kFinalLayerWeightsAddress = 1;
  CHECK_EQ(
      kFinalLayerWeightsAddress,
      Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress);
  constexpr AddressT kFirstLayerOutputBeforeReluAddress = 2;
  constexpr AddressT kFirstLayerOutputAfterReluAddress = 3;
  constexpr AddressT kZerosAddress = 4;
  constexpr AddressT kGradientWrtFinalLayerWeightsAddress = 5;
  constexpr AddressT kGradientWrtActivationsAddress = 6;
  constexpr AddressT kGradientOfReluAddress = 7;
  CHECK_GE(kMaxVectorAddresses, 8);

  // Matrix addresses.
  constexpr AddressT kFirstLayerWeightsAddress = 0;
  CHECK_EQ(
      kFirstLayerWeightsAddress,
      Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress);
  constexpr AddressT kGradientWrtFirstLayerWeightsAddress = 1;
  CHECK_GE(kMaxMatrixAddresses, 2);

  shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();

  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kLearningRateAddress,
      ActivationDataSetter(learning_rate)));
  PadComponentFunctionWithInstruction(
      setup_size_init_, no_op_instruction, &algorithm.setup_);

  IntegerT num_predict_instructions = 5;
  algorithm.predict_.reserve(num_predict_instructions);
  // Multiply with first layer weight matrix.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_VECTOR_PRODUCT_OP,
      kFirstLayerWeightsAddress, kFeaturesVectorAddress,
      kFirstLayerOutputBeforeReluAddress));
  // Apply RELU.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_MAX_OP, kFirstLayerOutputBeforeReluAddress, kZerosAddress,
      kFirstLayerOutputAfterReluAddress));
  // Dot product with final layer weight vector.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP, kFirstLayerOutputAfterReluAddress,
      kFinalLayerWeightsAddress, kPredictionsScalarAddress));
  PadComponentFunctionWithInstruction(
      predict_size_init_, no_op_instruction, &algorithm.predict_);

  algorithm.learn_.reserve(11);
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP, kLabelsScalarAddress, kPredictionsScalarAddress,
      kPredictionErrorAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      kLearningRateAddress, kPredictionErrorAddress, kPredictionErrorAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP, kPredictionErrorAddress,
      kFirstLayerOutputAfterReluAddress, kGradientWrtFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      kFinalLayerWeightsAddress, kGradientWrtFinalLayerWeightsAddress,
      kFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      kPredictionErrorAddress, kFinalLayerWeightsAddress,
      kGradientWrtActivationsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_HEAVYSIDE_OP,
      kFirstLayerOutputBeforeReluAddress, 0, kGradientOfReluAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_PRODUCT_OP,
      kGradientOfReluAddress, kGradientWrtActivationsAddress,
      kGradientWrtActivationsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_OUTER_PRODUCT_OP,
      kGradientWrtActivationsAddress, kFeaturesVectorAddress,
      kGradientWrtFirstLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_SUM_OP,
      kFirstLayerWeightsAddress, kGradientWrtFirstLayerWeightsAddress,
      kFirstLayerWeightsAddress));
  PadComponentFunctionWithInstruction(
      learn_size_init_, no_op_instruction, &algorithm.learn_);

  return algorithm;
}

Algorithm Generator::My_Algorithm() {
  Algorithm algorithm;

  constexpr AddressT kFirstFeature = 2; 
  constexpr AddressT kSecondFeature = 3; 
  constexpr AddressT kThirdFeature = 4; 
  constexpr AddressT kForthFeature = 5; 
  constexpr AddressT kFifthFeature = 6; 
  constexpr AddressT kSixthFeature = 7; 
  constexpr AddressT kSeventhFeature = 8; 
  constexpr AddressT kEighthFeature = 9; 
  constexpr AddressT kNinthFeature = 10; 
  constexpr AddressT kTenthFeature = 11;  
  constexpr AddressT kEleventhFeature = 12; 
  constexpr AddressT kTwelfthFeature = 13; 
  constexpr AddressT kThirteenthFeature = 14;  
  constexpr AddressT kAlphaAnominator = 15;  
  constexpr AddressT kAlphaDenominator = 16;
  CHECK_GE(kMaxScalarAddresses, 17);

  constexpr AddressT kFirstElementAddress = 1;
  CHECK_GE(kMaxVectorAddresses, 2);

  CHECK_GE(kMaxMatrixAddresses, 0);

  shared_ptr<const Instruction> no_op_instruction =
    make_shared<const Instruction>();

  FillComponentFunctionWithInstruction(
    setup_size_init_, no_op_instruction, &algorithm.setup_);

  for (double i=0.1; i < 13; i++) {
    if (0 < i && i < 1) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i), FloatDataSetter(1.0)));         
    }
    else
    {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i), FloatDataSetter(0.0)));  
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kFirstFeature));
  for (double i=0.1; i < 13; i++) {
    if (0 < i && i < 1) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (1 < i && i < 2) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kSecondFeature));
  for (double i=0.1; i < 13; i++) {
    if (1 < i && i < 2) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (2 < i && i < 3) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kThirdFeature));
  for (double i=0.1; i < 13; i++) {
    if (2 < i && i < 3) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (3 < i && i < 4) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kForthFeature));
  for (double i=0.1; i < 13; i++) {
    if (3 < i && i < 4) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (4 < i && i < 5) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kFifthFeature));
  for (double i=0.1; i < 13; i++) {
    if (4 < i && i < 5) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (5 < i && i < 6) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kSixthFeature));
  for (double i=0.1; i < 13; i++) {
    if (5 < i && i < 6) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (6 < i && i < 7) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kSeventhFeature));
  for (double i=0.1; i < 13; i++) {
    if (6 < i && i < 7) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (7 < i && i < 8) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kEighthFeature));
  for (double i=0.1; i < 13; i++) {
    if (7 < i && i < 8) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (8 < i && i < 9) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kNinthFeature));
  for (double i=0.1; i < 13; i++) {
    if (8 < i && i < 9) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (9 < i && i < 10) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kTenthFeature));
  for (double i=0.1; i < 13; i++) {
    if (9 < i && i < 10) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (10 < i && i < 11) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kEleventhFeature));
  for (double i=0.1; i < 13; i++) {
    if (10 < i && i < 11) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (11 < i && i < 12) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kTwelfthFeature));
  for (double i=0.1; i < 13; i++) {
    if (11 < i && i < 12) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
    if (12 < i && i < 13) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(1.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
    VECTOR_INNER_PRODUCT_OP,
    kFirstElementAddress, kFeaturesMatrixAddress, kThirteenthFeature));
  for (double i=0.1; i < 13; i++) {
    if (12 < i && i < 13) {
      algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      kFirstElementAddress, FloatDataSetter(i/13), FloatDataSetter(0.0)));         
    }
  }
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP,
      kTwelfthFeature, kNinthFeature, kAlphaAnominator));
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP,
      kTenthFeature, kEleventhFeature, kAlphaDenominator));
  // Apply RELU.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIVISION_OP, kAlphaAnominator, kAlphaDenominator, 
      kPredictionsScalarAddress));
  PadComponentFunctionWithInstruction(
      predict_size_init_, no_op_instruction, &algorithm.predict_);
  FillComponentFunctionWithInstruction(
    learn_size_init_, no_op_instruction_, &algorithm.learn_);

  return algorithm;
}

template<typename T>
bool Generator::isNumber(T x){
    std::string s;
    std::string f="5";
    std::regex e ("^-?\\d*\\.?\\d+");
    std::stringstream ss; 
    ss << x;
    ss >> s >> f;
    
    char* p;
    const char* pp = s.c_str();
    double d1=strtod(pp, &p);

    char* p2;
    const char* ppp = f.c_str();
    double d2=strtod(ppp, &p2);
//    if ((std::regex_match (s,e) && std::regex_match (f,e)) != ((*p == 0 || isspace(*p)) && (*p2 == 0 || isspace(*p2))))
//    {
//    std::cout << "std::regex_match (s,e): " << std::regex_match (s,e) << std::endl;
//    std::cout << "std::regex_match (f,e): " << std::regex_match (f,e) << std::endl;
//    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << d1 << std::endl;
//    std::cout << "input: " << x << std::endl;
//    std::cout << "d1: " << d1 << std::endl;
//    if (p) {std::cout << "it is not a null ptr" << std::endl;}
//    else std::cout << "it is a null ptr" << std::endl;
//    if (*p == 0 || isspace(*p)) {std::cout << "it is number by p " << std::endl;}
//    else std::cout << "it is not number by p " << std::endl;}

    if ((*p == 0 || isspace(*p)) && (*p2 == 0 || isspace(*p2))) return true;
    else return false;

    //if (std::regex_match (s,e) && std::regex_match (f,e)) return true;
    //else return false;
}

AddressT Generator::AddressTer(const std::string variable_name) {
  IntegerT add_num = 1234; 
  string variable_num_str = variable_name.substr(1, variable_name.length());
  if (variable_name.substr(0, 1) == "s") {
    add_num = std::stod(&variable_num_str[0]); 
    CHECK(add_num < 30);
  } else if (variable_name.substr(0, 1) == "v") {
    add_num = std::stod(&variable_num_str[0]) + 30; 
    CHECK((add_num - 30) < 30);
  } else if (variable_name.substr(0, 1) == "m") {
    add_num = std::stod(&variable_num_str[0]) + 60;
    CHECK((add_num - 60) < 25); 
  } else {
    std::cout << "error!!!" << std::endl;
  }
  switch (add_num) {
    case 0: {
      return kLabelsScalarAddress;
    }   
    case 1: {
      return kPredictionsScalarAddress;
    }    
    case 2: {
      constexpr AddressT kInitS2 = 2;
      return kInitS2;
    }
    case 3: {
      constexpr AddressT kInitS3 = 3;
      return kInitS3;
    }
    case 4: {
      constexpr AddressT kInitS4 = 4;
      return kInitS4;
    }
    case 5: {
      constexpr AddressT kInitS5 = 5;
      return kInitS5;
    }
    case 6: {
      constexpr AddressT kInitS6 = 6;
      return kInitS6;
    }
    case 7: {
      constexpr AddressT kInitS7 = 7;
      return kInitS7;
    }
    case 8: {
      constexpr AddressT kInitS8 = 8;
      return kInitS8;
    }
    case 9: {
      constexpr AddressT kInitS9 = 9;
      return kInitS9;
    }
    case 10: {
      constexpr AddressT kInitS10 = 10;
      return kInitS10;
    }
    case 11: {
      constexpr AddressT kInitS11 = 11;
      return kInitS11;
    }
    case 12: {
      constexpr AddressT kInitS12 = 12;
      return kInitS12;
    }
    case 13: {
      constexpr AddressT kInitS13 = 13;
      return kInitS13;
    }
    case 14: {
      constexpr AddressT kInitS14 = 14;
      return kInitS14;
    }
    case 15: {
      constexpr AddressT kInitS15 = 15;
      CHECK_GE(kMaxScalarAddresses, 15);
      return kInitS15;
    }
    case 16: {
      constexpr AddressT kInitS16 = 16;
      CHECK_GE(kMaxScalarAddresses, 16);
      return kInitS16;
    }
    case 17: {
      constexpr AddressT kInitS17 = 17;
      CHECK_GE(kMaxScalarAddresses, 17);
      return kInitS17;
    }
    case 18: {
      constexpr AddressT kInitS18 = 18;
      CHECK_GE(kMaxScalarAddresses, 18);
      return kInitS18;
    }
    case 19: {
      constexpr AddressT kInitS19 = 19;
      CHECK_GE(kMaxScalarAddresses, 19);
      return kInitS19;
    }
    case 20: {
      constexpr AddressT kInits20 = 20;
      return kInits20;
    }  
    case 21: {
      constexpr AddressT kInits21 = 21;
      return kInits21;
    }    
    case 22: {
      constexpr AddressT kInits22 = 22;
      return kInits22;
    }
    case 23: {
      constexpr AddressT kInits23 = 23;
      return kInits23;
    }
    case 24: {
      constexpr AddressT kInits24 = 24;
      return kInits24;
    }
    case 25: {
      constexpr AddressT kInits25 = 25;
      return kInits25;
    }
    case 26: {
      constexpr AddressT kInits26 = 26;
      return kInits26;
    }
    case 27: {
      constexpr AddressT kInits27 = 27;
      CHECK_GE(kMaxScalarAddresses, 27);
      return kInits27;
    }
    case 28: {
      constexpr AddressT kInits28 = 28;
      CHECK_GE(kMaxScalarAddresses, 28);
      return kInits28;
    }
    case 29: {
      constexpr AddressT kInits29 = 29;
      CHECK_GE(kMaxScalarAddresses, 29);
      return kInits29;
    }
    case 30: {
      constexpr AddressT kInitv0 = 0;
      return kInitv0;
    }  
    case 31: {
      constexpr AddressT kInitv1 = 1;
      return kInitv1;
    }    
    case 32: {
      constexpr AddressT kInitv2 = 2;
      return kInitv2;
    }
    case 33: {
      constexpr AddressT kInitv3 = 3;
      return kInitv3;
    }
    case 34: {
      constexpr AddressT kInitv4 = 4;
      return kInitv4;
    }
    case 35: {
      constexpr AddressT kInitv5 = 5;
      return kInitv5;
    }
    case 36: {
      constexpr AddressT kInitv6 = 6;
      return kInitv6;
    }
    case 37: {
      constexpr AddressT kInitv7 = 7;
      return kInitv7;
    }
    case 38: {
      constexpr AddressT kInitv8 = 8;
      return kInitv8;
    }
    case 39: {
      constexpr AddressT kInitv9 = 9;
      return kInitv9;
    }
    case 40: {
      constexpr AddressT kInitv10 = 10;
      return kInitv10;
    }
    case 41: {
      constexpr AddressT kInitv11 = 11;
      return kInitv11;
    }
    case 42: {
      constexpr AddressT kInitv12 = 12;
      CHECK_GE(kMaxVectorAddresses, 12);
      return kInitv12;
    }
    case 43: {
      constexpr AddressT kInitv13 = 13;
      CHECK_GE(kMaxVectorAddresses, 13);
      return kInitv13;
    }
    case 44: {
      constexpr AddressT kInitv14 = 14;
      CHECK_GE(kMaxVectorAddresses, 14);
      return kInitv14;
    }
    case 45: {
      constexpr AddressT kInitv15 = 15;
      CHECK_GE(kMaxVectorAddresses, 15);
      return kInitv15;
    }
    case 46: {
      constexpr AddressT kInitv16 = 16;
      CHECK_GE(kMaxVectorAddresses, 16);
      return kInitv16;
    }
    case 47: {
      constexpr AddressT kInitv17 = 17;
      CHECK_GE(kMaxVectorAddresses, 17);
      return kInitv17;
    }
    case 48: {
      constexpr AddressT kInitv18 = 18;
      CHECK_GE(kMaxVectorAddresses, 18);
      return kInitv18;
    }
    case 49: {
      constexpr AddressT kInitv19 = 19;
      CHECK_GE(kMaxVectorAddresses, 19);
      return kInitv19;
    }
    case 50: {
      constexpr AddressT kInitv20 = 20;
      CHECK_GE(kMaxVectorAddresses, 20);
      return kInitv20;
    }
    case 51: {
      constexpr AddressT kInitv21 = 21;
      CHECK_GE(kMaxVectorAddresses, 21);
      return kInitv21;
    }    
    case 52: {
      constexpr AddressT kInitv22 = 22;
      CHECK_GE(kMaxVectorAddresses, 22);
      return kInitv22;
    }
    case 53: {
      constexpr AddressT kInitv23 = 23;
      CHECK_GE(kMaxVectorAddresses, 23);
      return kInitv23;
    }
    case 54: {
      constexpr AddressT kInitv24 = 24;
      CHECK_GE(kMaxVectorAddresses, 24);
      return kInitv24;
    }
    case 55: {
      constexpr AddressT kInitv25 = 25;
      CHECK_GE(kMaxVectorAddresses, 25);
      return kInitv25;
    }
    case 56: {
      constexpr AddressT kInitv26 = 26;
      CHECK_GE(kMaxVectorAddresses, 26);
      return kInitv26;
    }
    case 57: {
      constexpr AddressT kInitv27 = 27;
      CHECK_GE(kMaxVectorAddresses, 27);
      return kInitv27;
    }
    case 58: {
      constexpr AddressT kInitv28 = 28;
      CHECK_GE(kMaxVectorAddresses, 28);
      return kInitv28;
    }
    case 59: {
      constexpr AddressT kInitv29 = 29;
      CHECK_GE(kMaxVectorAddresses, 29);
      return kInitv29;
    }
    case 60: {
      return kFeaturesMatrixAddress;
    }
    case 61: {
      constexpr AddressT kInitm1 = 1;
      CHECK_GE(kMaxMatrixAddresses, 1);
      return kInitm1;
    }    
    case 62: {
      constexpr AddressT kInitm2 = 2;
      CHECK_GE(kMaxMatrixAddresses, 2);
      return kInitm2;
    }
    case 63: {
      constexpr AddressT kInitm3 = 3;
      CHECK_GE(kMaxMatrixAddresses, 3);
      return kInitm3;
    }
    case 64: {
      constexpr AddressT kInitm4 = 4;
      CHECK_GE(kMaxMatrixAddresses, 4);
      return kInitm4;
    }
    case 65: {
      constexpr AddressT kInitm5 = 5;
      CHECK_GE(kMaxMatrixAddresses, 5);
      return kInitm5;
    }
    case 66: {
      constexpr AddressT kInitm6 = 6;
      CHECK_GE(kMaxMatrixAddresses, 6);
      return kInitm6;
    }
    case 67: {
      constexpr AddressT kInitm7 = 7;
      CHECK_GE(kMaxMatrixAddresses, 7);
      return kInitm7;
    }
    case 68: {
      constexpr AddressT kInitm8 = 8;
      CHECK_GE(kMaxMatrixAddresses, 8);
      return kInitm8;
    }
    case 69: {
      constexpr AddressT kInitm9 = 9;
      CHECK_GE(kMaxMatrixAddresses, 9);
      return kInitm9;
    }
    case 70: {
      constexpr AddressT kInitm10 = 10;
      CHECK_GE(kMaxMatrixAddresses, 10);
      return kInitm10;
    }
    case 71: {
      constexpr AddressT kInitm11 = 11;
      return kInitm11;
    }
    case 72: {
      constexpr AddressT kInitm12 = 12;
      return kInitm12;
    }
    case 73: {
      constexpr AddressT kInitm13 = 13;
      return kInitm13;
    }
    case 74: {
      constexpr AddressT kInitm14 = 14;
      return kInitm14;
    }
    case 75: {
      constexpr AddressT kInitm15 = 15;
      return kInitm15;
    }
    case 76: {
      constexpr AddressT kInitm16 = 16;
      return kInitm16;
    }
    case 77: {
      constexpr AddressT kInitm17 = 17;
      return kInitm17;
    }
    case 78: {
      constexpr AddressT kInitm18 = 18;
      return kInitm18;
    }
    case 79: {
      constexpr AddressT kInitm19 = 19;
      return kInitm19;
    }
    case 80: {
      constexpr AddressT kInitm20 = 20;
      return kInitm20;
    }
    case 81: {
      constexpr AddressT kInitm21 = 21;
      return kInitm21;
    }
    case 82: {
      constexpr AddressT kInitm22 = 22;
      return kInitm22;
    }
    case 83: {
      constexpr AddressT kInitm23 = 23;
      return kInitm23;
    }
    case 84: {
      constexpr AddressT kInitm24 = 24;
      return kInitm24;
    }
    case 85: {
      constexpr AddressT kInitm25 = 25;
      return kInitm25;
    }
    default:
      LOG(FATAL) << "Unsupported variable name." << endl;
  }
}

void Generator::BuildFromInput(Algorithm* algorithm, IntegerT current_part, IntegerT instruction_num, double int1, double int2, double assigned_double, string variable, string variable_int1, string variable_int2, string variable_int3, string variable_int4) {

  switch (current_part) {
    case 0: {
      EmplaceSetup(algorithm, instruction_num, int1, int2, assigned_double, variable_int1, variable_int2, variable); /// I put variable at the end; so later in the funtion of emplacepredict variable2 is variable. The reason is that out variable is read first but in instruction.cc out variable is placed at the back
      return;
    }
    case 1: {
      EmplacePredict(algorithm, instruction_num, int1, int2, assigned_double, variable_int1, variable_int2, variable, variable_int3, variable_int4); /// I put variable at the end; so later in the funtion of emplacepredict variable2 is variable. The reason is that out variable is read first but in instruction.cc out variable is placed at the back
      return;
    }
    case 2: {   
      EmplaceLearn(algorithm, instruction_num, int1, int2, assigned_double, variable_int1, variable_int2, variable, variable_int3, variable_int4); /// I put variable at the end; so later in the funtion of emplacepredict variable2 is variable. The reason is that out variable is read first but in instruction.cc out variable is placed at the back
      return;
    }
  }
}

void Generator::EmplaceSetup(Algorithm* algorithm, IntegerT instruction_num, double int1, double int2, double assigned_double, string variable, string variable_int1, string variable_int2) {
  switch (instruction_num) {
    case 1: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 2: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 3: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 4: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 44: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 47: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 5: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 15: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 56: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      AddressTer(variable_int2),
      ActivationDataSetter(assigned_double)));
      return;
    }
    case 6: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 7: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_SIN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 8: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_COS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 9: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_TAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 10: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCSIN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 11: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCCOS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 12: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCTAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 13: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_EXP_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 14: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_LOG_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 23: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 24: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 25: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 26: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 45: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 48: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 22: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 16: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 57: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(assigned_double)));
      return;
    }
    case 20: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 39: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 40: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 41: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 42: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 46: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 49: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 38: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 17: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 58: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_CONST_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2),
      FloatDataSetter(assigned_double)));
      return;
    }
    case 30: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 18: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 27: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 28: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_OUTER_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 29: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 31: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 21: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 34: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }    
    case 36: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 35: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_COLUMN_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 37: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_TRANSPOSE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 43: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 19: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 32: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_COLUMN_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 33: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_ROW_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 50: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 54: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 51: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 55: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 52: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 53: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 62: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 63: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 64: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 59: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 60: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 61: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 67: {
      algorithm->setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_SCALAR_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 0: {
      shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();
      PadComponentFunctionWithInstruction(
      1, no_op_instruction, &algorithm->setup_);
      return;
    }
    default:
      LOG(FATAL) << "Unsupported variable name." << endl;
  }
}

void Generator::EmplacePredict(Algorithm* algorithm, IntegerT instruction_num, double int1, double int2, double assigned_double, string variable, string variable_int1, string variable_int2, string variable_int3, string variable_int4) {
  switch (instruction_num) {
    case 1: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 2: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 3: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 4: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 44: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 47: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 5: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 15: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 56: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      AddressTer(variable_int2),
      ActivationDataSetter(assigned_double)));
      return;
    }
    case 6: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 7: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_SIN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 8: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_COS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 9: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_TAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 10: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCSIN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 11: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCCOS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 12: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCTAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 13: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_EXP_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 14: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_LOG_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 23: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 24: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 25: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 26: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 45: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 48: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 22: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 16: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 57: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(assigned_double)));
      return;
    }
    case 20: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 39: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 40: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 41: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 42: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 46: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 49: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 38: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 17: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 58: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_CONST_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2),
      FloatDataSetter(assigned_double)));
      return;
    }
    case 30: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 18: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 27: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 28: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_OUTER_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 29: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 31: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }      
    case 21: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 34: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }    
    case 36: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 35: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_COLUMN_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 37: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_TRANSPOSE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 43: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 19: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 32: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_COLUMN_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 33: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_ROW_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 50: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 54: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 51: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 55: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 52: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 53: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 62: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 63: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 64: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 59: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 60: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 61: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 67: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_SCALAR_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 66: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_PREVIOUS_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 65: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_RELATION_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 75: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      RELATION_DEMEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 68: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_COLUMN_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 69: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_ROW_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 70: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      CONDITION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int3),
      AddressTer(variable_int4),
      AddressTer(variable_int2)));
      return;
    }
    case 71: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      CORRELATION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 72: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      TS_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 73: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      TS_ROW_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 74: {
      algorithm->predict_.emplace_back(make_shared<const Instruction>(
      COVARIANCE_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 0: {
      shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();
      PadComponentFunctionWithInstruction(
      1, no_op_instruction, &algorithm->predict_);
      return;
    }
    default:
      LOG(FATAL) << "Unsupported variable name." << endl;
  }
}

void Generator::EmplaceLearn(Algorithm* algorithm, IntegerT instruction_num, double int1, double int2, double assigned_double, string variable, string variable_int1, string variable_int2, string variable_int3, string variable_int4) {
  switch (instruction_num) {
    case 1: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 2: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 3: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 4: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 44: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 47: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 5: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 15: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 56: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      AddressTer(variable_int2),
      ActivationDataSetter(assigned_double)));
      return;
    }
    case 6: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 7: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_SIN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 8: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_COS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 9: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_TAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 10: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCSIN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 11: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCCOS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 12: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_ARCTAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 13: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_EXP_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 14: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_LOG_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 23: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 24: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 25: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 26: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 45: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 48: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 22: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 16: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 57: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_CONST_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(assigned_double)));
      return;
    }
    case 20: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 39: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_SUM_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 40: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_DIFF_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 41: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 42: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_DIVISION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 46: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_MIN_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 49: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_MAX_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 38: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_ABS_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 17: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_HEAVYSIDE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 58: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_CONST_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2),
      FloatDataSetter(assigned_double)));
      return;
    }
    case 30: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_RECIPROCAL_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 18: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 27: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 28: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_OUTER_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 29: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 31: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_VECTOR_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }    
    case 21: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 34: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }    
    case 36: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 35: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_COLUMN_NORM_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 37: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_TRANSPOSE_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 43: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_MATRIX_PRODUCT_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2)));
      return;
    }
    case 19: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 32: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_COLUMN_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 33: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_ROW_BROADCAST_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 50: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 54: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 51: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 55: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 52: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_MEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 53: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_ST_DEV_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 62: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 63: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 64: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_GAUSSIAN_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 59: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 60: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 61: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_UNIFORM_SET_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 68: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_COLUMN_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 69: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_ROW_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 70: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      CONDITION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int3),
      AddressTer(variable_int4),
      AddressTer(variable_int2)));
      return;
    }
    case 71: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      CORRELATION_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 72: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      TS_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 73: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      TS_ROW_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 74: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      COVARIANCE_OP,
      AddressTer(variable),
      AddressTer(variable_int1),
      AddressTer(variable_int2),
      FloatDataSetter(int1)));
      return;
    }
    case 67: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_SCALAR_OP,
      AddressTer(variable_int2),
      FloatDataSetter(int1),
      FloatDataSetter(int2)));
      return;
    }
    case 66: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_PREVIOUS_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 65: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_RELATION_RANK_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 75: {
      algorithm->learn_.emplace_back(make_shared<const Instruction>(
      RELATION_DEMEAN_OP,
      AddressTer(variable),
      AddressTer(variable_int2)));
      return;
    }
    case 0: {
      shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();
      PadComponentFunctionWithInstruction(
      1, no_op_instruction, &algorithm->learn_);
      return;
    }
    default:
      LOG(FATAL) << "Unsupported variable name." << endl;
  }
}

Algorithm Generator::MyNeuralNet() {
  Algorithm algorithm;

  // Scalar addresses
  constexpr AddressT kInitS3 = 2;
  constexpr AddressT kLearningRateAddress = 3;
  constexpr AddressT kPredictionErrorAddress = 4;
  CHECK_GE(kMaxScalarAddresses, 5);

  // Vector addresses.
  constexpr AddressT kInitV3 = 1;
  constexpr AddressT kFinalLayerWeightsAddress = 2;
  constexpr AddressT kFirstLayerOutputBeforeReluAddress = 3;
  constexpr AddressT kFirstLayerOutputAfterReluAddress = 4;
  constexpr AddressT kZerosAddress = 5;
  constexpr AddressT kGradientWrtFinalLayerWeightsAddress = 6;
  constexpr AddressT kGradientWrtActivationsAddress = 7;
  constexpr AddressT kGradientOfReluAddress = 8;
  CHECK_GE(kMaxVectorAddresses, 9);

  // Matrix addresses.
  constexpr AddressT kFirstLayerWeightsAddress = 0;
  constexpr AddressT kGradientWrtFirstLayerWeightsAddress = 1;
  CHECK_GE(kMaxMatrixAddresses, 2);

  shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();

  IntegerT current_part = 1234;
  for (auto line: read_from_file_)
  {
      std::string delimiterEq = "=";
      std::string delimiterCo = ",";
      std::string delimiterSb = "[";
      std::string delimiterSbr = "]";
      std::string delimiterPl = "+";
      std::string delimiterMi = "-";
      std::string delimiterMu = "*";
      std::string delimiterDi = "/";
      std::string delimiterBl = "(";
      std::string delimiterBr = ")";
      std::string delimiterSp = " ";
      std::string delimiterCl = ":";
      std::string delimiterQs = "?";
      std::string delimiterGt = ">";
      double int1, int2, assigned_double;
      string instruction, variable, variable_int1, variable_int2;
      IntegerT instruction_num;
      if (line.find(delimiterCl) != std::string::npos) {
      if (line.substr(0, line.find(delimiterCl)) == "def Setup()") {
        current_part = 0;}
      else if (line.substr(0, line.find(delimiterCl)) == "def Predict()") {
        PadComponentFunctionWithInstruction(
          setup_size_init_, no_op_instruction, &algorithm.setup_);  
          current_part = 1;}
      else if (line.substr(0, line.find(delimiterCl)) == "def Learn()") {
        PadComponentFunctionWithInstruction(
          predict_size_init_, no_op_instruction, &algorithm.predict_);  
          current_part = 2;}
        }
      if (line.substr(0, 6) == "NoOp()") {
          instruction = "NoOp";
          instruction_num = 0;
          BuildFromInput(&algorithm, current_part, instruction_num, int1, int2, assigned_double, variable, variable_int1, variable_int2);
        }
      if (line.find(delimiterEq) != std::string::npos) {
        std::string after_eq = line.substr(line.find(delimiterEq)+2, line.length()); // +2 to get no space;
        std::string before_eq = line.substr(0, line.find(delimiterEq)-1); // -1 to get no space
        std::string instruction_front = ""; // if not changed then error
        if (before_eq.substr(0, 1) == "m") {
          instruction_front = "matrix";
        } 
        else if (before_eq.substr(0, 1) == "v") {
            instruction_front = "vector";
          } 
        else if (before_eq.substr(0, 1) == "s") {
            instruction_front = "scalar";
          } 
        else {
            LOG(FATAL) << "Unsupported instruction front name." << endl;
            CHECK(before_eq.substr(0, 1) == "gg");
          }

          /// for condition op
        if (line.find(delimiterQs) != std::string::npos) {
          variable_int1 = after_eq.substr(line.find(delimiterEq)+2, line.find(delimiterGt)-1-(line.find(delimiterEq)+2)); 
          variable_int1 = after_eq.substr(line.find(delimiterGt)+2, line.find(delimiterQs)-1-(line.find(delimiterGt)+2));
          variable_int1 = after_eq.substr(line.find(delimiterQs)+2, line.find(delimiterCl)-1-(line.find(delimiterQs)+2));
          variable_int1 = after_eq.substr(line.find(delimiterCl)+2, line.length());
          instruction_num = 70;
          instruction = "condition";
        }          

        // const_set_op
        if (isNumber(after_eq)) { // check if a float is after the eq
          assigned_double = std::stod(&after_eq[0]); 
          variable = before_eq.substr(0, before_eq.find(delimiterSb));

          // get numbers before the eq for const set op
          if (before_eq.find(delimiterSb) != std::string::npos) {
          std::string before_eq_after_Sb = before_eq.substr(before_eq.find(delimiterSb)+1, before_eq.find(delimiterSbr) - (before_eq.find(delimiterSb)+1));
            if (before_eq_after_Sb.find(delimiterCo) != std::string::npos) {
              std::string before_eq_after_Sb_before_comma = before_eq_after_Sb.substr(0, before_eq_after_Sb.find(delimiterCo));
              int1 = std::stod(&before_eq_after_Sb_before_comma[0]); 
              std::string before_eq_after_Sb_after_comma = before_eq_after_Sb.substr(before_eq_after_Sb.find(delimiterCo)+2, before_eq_after_Sb.length());
              int2 = std::stod(&before_eq_after_Sb_after_comma[0]); 
            } else {
              int1 = std::stod(&before_eq_after_Sb[0]); 
            }
          }
          instruction = "_const_set";
          if (instruction_front == "scalar") {
            instruction_num = 56;
          } else if (instruction_front == "vector") {
            instruction_num = 57;
          } else if (instruction_front == "matrix") {
            instruction_num = 58;
          }
        } else {
          variable = before_eq;
          if ((after_eq.find(delimiterBl)) != std::string::npos) {
            instruction = after_eq.substr(0, after_eq.find(delimiterBl));
            std::string after_eq_before_bl = after_eq.substr(0, after_eq.find(delimiterBl));

            if (after_eq_before_bl == "get_scalar") {
              instruction = "get_scalar";
              instruction_num = 67; 
            }
            if (after_eq_before_bl == "get_column") {
              instruction = "get_column";
              instruction_num = 68; 
            }
            if (after_eq_before_bl == "get_row") {
              instruction = "get_row";
              instruction_num = 69; 
            }
            if (after_eq_before_bl == "correlation") {
              instruction = "correlation";
              instruction_num = 71; 
            }
            if (after_eq_before_bl == "covariance") {
              instruction = "covariance";
              instruction_num = 74; 
            }
            if (after_eq_before_bl == "previous_rank") {
              instruction = "previous_rank";
              instruction_num = 66; 
            }
            if (after_eq_before_bl == "relation_rank") {
              instruction = "relation_rank";
              instruction_num = 65; 
            }
            if (after_eq_before_bl == "relation_demean") {
              instruction = "relation_demean";
              instruction_num = 75; 
            }
            if (after_eq_before_bl == "TS_rank") {
              instruction = "TS_rank";
              instruction_num = 72; 
            }
            if (after_eq_before_bl == "TS_row_rank") {
              instruction = "TS_row_rank";
              instruction_num = 73; 
            }
            // extract numbers in1/in2 or variable int1/in2
            std::string after_eq_after_bl = after_eq.substr(after_eq.find(delimiterBl)+1, after_eq.find(delimiterBr)-(after_eq.find(delimiterBl)+1));

            // if comma exists after eq then is gaussian const set or other variable int1/int2
            if (after_eq_after_bl.find(delimiterCo) != std::string::npos) {
              // before comma
              std::string after_eq_after_bl_before_co = after_eq_after_bl.substr(0, after_eq_after_bl.find(delimiterCo));
              if (after_eq_after_bl_before_co.length() > 3) {
                CHECK(isNumber(after_eq_after_bl_before_co));
                int1 = std::stod(&after_eq_after_bl_before_co[0]); }
              else {variable_int1 = after_eq_after_bl_before_co;}
              // after comma
              std::string after_eq_after_bl_after_co = after_eq_after_bl.substr(after_eq_after_bl.find(delimiterCo)+2, after_eq_after_bl.length()); 

              if (after_eq_before_bl == "covariance" || after_eq_before_bl == "correlation") {
                std::string after_eq_after_bl_after_co_after_co = after_eq_after_bl_after_co.substr(after_eq_after_bl_after_co.find(delimiterCo)+2, after_eq_after_bl_after_co.length()); 
                int1 = std::stod(&after_eq_after_bl_after_co_after_co[0]);
              }

              // if after comma length larger than 3 then could be float could be axis
              if (isNumber(after_eq_after_bl_after_co.substr(0, 5)) && after_eq_after_bl_after_co.substr(0, 4) != "axis") {
                int2 = std::stod(&after_eq_after_bl_after_co[0]);
              }
              // after comma <=3 then it's a variable name
              else if (!isNumber(after_eq_after_bl_after_co.substr(0, 5))) {
                variable_int2 = after_eq_after_bl_after_co.substr(0, after_eq_after_bl_after_co.length());
                // matrix row mean op
                if (variable_int1.substr(0, 1) == "m" && instruction == "mean" && after_eq_after_bl_after_co.substr(0, 6) == "axis=1") {
                  instruction = "matrix_row_mean";
                  instruction_num = 52;
                }
                // matrix row std op
                else if (variable_int1.substr(0, 1) == "m" && instruction == "std" && after_eq_after_bl_after_co.substr(0, 6) == "axis=1") {
                  instruction = "matrix_row_std";
                  instruction_num = 53;
                }    
                // vector column broadcast op
                else if (variable_int1.substr(0, 1) == "v" && instruction == "bcast" && after_eq_after_bl_after_co.substr(0, 6) == "axis=0") {
                  instruction = "vector_column_broadcast";
                  instruction_num = 32;
                }   
                // vector row broadcast op
                else if (variable_int1.substr(0, 1) == "v" && instruction == "bcast" && after_eq_after_bl_after_co.substr(0, 6) == "axis=1") {
                  instruction = "vector_row_broadcast";
                  instruction_num = 33;
                }
                // matrix column norm op
                else if (variable_int1.substr(0, 1) == "m" && instruction == "norm" && after_eq_after_bl_after_co.substr(0, 6) == "axis=0") {
                  instruction = "matrix_column_norm";
                  instruction_num = 35;
                }   
                // matrix row norm op
                else if (variable_int1.substr(0, 1) == "m" && instruction == "norm" && after_eq_after_bl_after_co.substr(0, 6) == "axis=1") {
                  instruction = "matrix_row_norm";
                  instruction_num = 36;
                }           
                // dot matrix vector product
                else if (instruction == "dot" && after_eq_after_bl_before_co.substr(0, 1) == "m" && after_eq_after_bl_after_co.substr(0, 1) == "v") {
                  instruction = "matrix_vector_product";
                  instruction_num = 31;
                }
                // dot vector inner product
                else if (instruction == "dot" && after_eq_after_bl_before_co.substr(0, 1) == "v" && after_eq_after_bl_after_co.substr(0, 1) == "v") {
                  instruction = "vector_inner_product";
                  instruction_num = 27;
                }
                // dot vector outer product
                else if (instruction == "outer" && after_eq_after_bl_before_co.substr(0, 1) == "v" && after_eq_after_bl_after_co.substr(0, 1) == "v") {
                  instruction = "vector_outer_product";
                  instruction_num = 28;
                }
                // dot vector outer product
                else if (instruction == "matmul" && after_eq_after_bl_before_co.substr(0, 1) == "m" && after_eq_after_bl_after_co.substr(0, 1) == "m") {
                  instruction_num = 43;
                }
              }
              // gaussian uniform set op
              if (instruction == "uniform") {
                if (instruction_front == "scalar") {
                  instruction_num = 59;
                }
                if (instruction_front == "vector") {
                  instruction_num = 60;
                }
                if (instruction_front == "matrix") {
                  instruction_num = 61;
                }
              }
              if (instruction == "gaussian") {
                if (instruction_front == "scalar") {
                  instruction_num = 62;
                }
                if (instruction_front == "vector") {
                  instruction_num = 63;
                }
                if (instruction_front == "matrix") {
                  instruction_num = 64;
                }
              }    
            } else {
              std::string after_eq_after_bl_no_co = after_eq_after_bl.substr(0, after_eq_after_bl.find(delimiterBr));
              if (!isNumber(after_eq_after_bl_no_co)) {
                variable_int1 = after_eq_after_bl_no_co;
                if (variable_int1.substr(0, 1) == "v" && instruction == "std") {
                  instruction = "vector_std";
                  instruction_num = 54;
                }
                // matrix std op
                if (variable_int1.substr(0, 1) == "m" && instruction == "std") {
                  instruction = "matrix_std";
                  instruction_num = 55;
                }
                // matrix std op
                if (variable_int1.substr(0, 1) == "m" && instruction == "norm") {
                  instruction = "matrix_norm";
                  instruction_num = 34;
                }
                // vector norm op
                if (variable_int1.substr(0, 1) == "v" && instruction == "norm") {
                  instruction = "vector_norm";
                  instruction_num = 21;
                }
                // scalar broadcast op
                if (variable_int1.substr(0, 1) == "s" && instruction == "bcast") {
                  instruction = "scalar_broadcast";
                  instruction_num = 19;
                }
                if (variable_int1.substr(0, 1) == "m" && instruction == "mean") {
                  instruction = "matrix_mean";
                  instruction_num = 51;
                }
                if (variable_int1.substr(0, 1) == "v" && instruction == "mean") {
                  instruction = "vector_mean";
                  instruction_num = 50;
                }                
              } else {
                int1 = std::stod(&after_eq_after_bl_no_co[0]); // for op get_column / get_row
              }
            }
            if (instruction_front == "scalar") {
              if (instruction == "abs") {
                instruction_num = 5;
              } else if (instruction == "reciprocal") {
                instruction_num = 6;
              } else if (instruction == "sin") {
                instruction_num = 7;
              } else if (instruction == "cos") {
                instruction_num = 8;
              } else if (instruction == "tan") {
                instruction_num = 9;
              } else if (instruction == "arcsin") {
                instruction_num = 10;
              } else if (instruction == "arccos") {
                instruction_num = 11;
              } else if (instruction == "arctan") {
                instruction_num = 12;
              } else if (instruction == "exp") {
                instruction_num = 13;
              } else if (instruction == "log") {
                instruction_num = 14;
              } else if (instruction == "heaviside") {
                instruction_num = 15;
              } else if (instruction == "minimum") {
                instruction_num = 44;
              } else if (instruction == "maximum") {
                instruction_num = 47;
              }
            } else if (instruction_front == "vector") {
              if (instruction == "heaviside") {
                instruction_num = 16;
              } else if (instruction == "reciprocal") {
                instruction_num = 20;
              } else if (instruction == "abs") {
                instruction_num = 22;
              } else if (instruction == "minimum") {
                instruction_num = 45;
              } else if (instruction == "maximum") {
                instruction_num = 48;
              } 
            } else if (instruction_front == "matrix") {
              if (instruction == "heaviside") {
                instruction_num = 17;
              } else if (instruction == "reciprocal") {
                instruction_num = 30;
              } else if (instruction == "transpose") {
                instruction_num = 37;
              } else if (instruction == "abs") {
                instruction_num = 38;
              } else if (instruction == "minimum") {
                instruction_num = 46;
              } else if (instruction == "maximum") {
                instruction_num = 49;
              }
            }
          }
          if ((after_eq.find(delimiterPl)) != std::string::npos) {
            instruction = instruction_front + "_" + "+";   
            // get in1/in2
            variable_int1 = after_eq.substr(0, after_eq.find(delimiterPl)-1);
            variable_int2 = after_eq.substr(after_eq.find(delimiterPl)+2, after_eq.length() - (after_eq.find(delimiterPl)+2));
            if (instruction_front == "scalar") {
              instruction_num = 1;
            } else if (instruction_front == "vector") {
              instruction_num = 23;
            } else if (instruction_front == "matrix") {
              instruction_num = 39;
            }
          }
            if ((after_eq.find(delimiterMi)) != std::string::npos && after_eq.substr((after_eq.find(delimiterMi)), 2)=="- ") {
              instruction = instruction_front + "_" + "-";
              // get in1/in2
              variable_int1 = after_eq.substr(0, after_eq.find(delimiterMi)-1);
              variable_int2 = after_eq.substr(after_eq.find(delimiterMi)+2, after_eq.length() - (after_eq.find(delimiterMi)+2));
              if (instruction_front == "scalar") {
                instruction_num = 2;
              } else if (instruction_front == "vector") {
                instruction_num = 24;
              } else if (instruction_front == "matrix") {
                instruction_num = 40;
              }
          }
            if ((after_eq.find(delimiterMu)) != std::string::npos) {
              // get in1/in2
              variable_int1 = after_eq.substr(0, after_eq.find(delimiterMu)-1);
              variable_int2 = after_eq.substr(after_eq.find(delimiterMu)+2, after_eq.length() - (after_eq.find(delimiterMu)+2));
              if (variable_int1.substr(0, 1) == "s" && variable_int2.substr(0, 1) == "s") {
                instruction = instruction_front + "_" + "*";
                instruction_num = 3;
              }          
              if (variable_int1.substr(0, 1) == "s" && variable_int2.substr(0, 1) == "v") {
                instruction = "scalar_vector_product";
                instruction_num = 18;
              }
              if (variable_int1.substr(0, 1) == "s" && variable_int2.substr(0, 1) == "m") {
                instruction = "scalar_matrix_product";
                instruction_num = 29;
              }
              if (variable_int1.substr(0, 1) == "v" && variable_int2.substr(0, 1) == "v") {
                instruction = instruction_front + "_" + "*";
                instruction_num = 25;
              }
              if (variable_int1.substr(0, 1) == "m" && variable_int2.substr(0, 1) == "m") {
                instruction = instruction_front + "_" + "*";
                instruction_num = 41;
              }
          }
            if ((after_eq.find(delimiterDi)) != std::string::npos) {
              // get in1/in2
              if (after_eq.substr(0, 1) == "1") {
                int1 = 1;
                variable_int1 = after_eq.substr(after_eq.find(delimiterDi)+2, after_eq.length() - (after_eq.find(delimiterDi)+2));
                if (instruction_front == "scalar") {
                  instruction_num = 6;
                } else if (instruction_front == "vector") {
                  instruction_num = 20;
                } else {
                  instruction_num = 30;
                }                
              } else {
                variable_int1 = after_eq.substr(0, after_eq.find(delimiterDi)-1);
                variable_int2 = after_eq.substr(after_eq.find(delimiterDi)+2, after_eq.length() - (after_eq.find(delimiterDi)+2));
                if (instruction_front == "scalar") {
                  instruction_num = 4;
                } else if (instruction_front == "vector") {
                  instruction_num = 26;
                } else {
                  instruction_num = 42;
                }
              }
          }    
        }
        BuildFromInput(&algorithm, current_part, instruction_num, int1, int2, assigned_double, variable, variable_int1, variable_int2);  
      }     
  }
  PadComponentFunctionWithInstruction(
    learn_size_init_, no_op_instruction, &algorithm.learn_); 
  return algorithm;
}

Algorithm Generator::NeuralNet(
    const double learning_rate,
    const double first_init_scale,
    const double final_init_scale) {
  Algorithm algorithm;

  // Scalar addresses
  constexpr AddressT kFinalLayerBiasAddress = 2;
  constexpr AddressT kLearningRateAddress = 3;
  constexpr AddressT kPredictionErrorAddress = 4;
  CHECK_GE(kMaxScalarAddresses, 5);

  // Vector addresses.
  constexpr AddressT kFirstLayerBiasAddress = 0;
  constexpr AddressT kFinalLayerWeightsAddress = 1;
  constexpr AddressT kFirstLayerOutputBeforeReluMatrixMeanAddress= 2;
  constexpr AddressT kFirstLayerOutputAfterReluAddress = 3;
  constexpr AddressT kZerosAddress = 4;
  constexpr AddressT kGradientWrtFinalLayerWeightsAddress = 5;
  constexpr AddressT kGradientWrtActivationsAddress = 6;
  constexpr AddressT kGradientOfReluAddress = 7;
  CHECK_GE(kMaxVectorAddresses, 8);

  // Matrix addresses.
  constexpr AddressT kFirstLayerWeightsAddress = 1;
  constexpr AddressT kGradientWrtFirstLayerWeightsAddress = 2;
  constexpr AddressT kFirstLayerOutputBeforeReluAddress = 3;
  CHECK_GE(kMaxMatrixAddresses, 4);

  shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();

  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_GAUSSIAN_SET_OP,
      kFinalLayerWeightsAddress,
      FloatDataSetter(0.0),
      FloatDataSetter(final_init_scale)));
  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_GAUSSIAN_SET_OP,
      kFirstLayerWeightsAddress,
      FloatDataSetter(0.0),
      FloatDataSetter(first_init_scale)));
  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kLearningRateAddress,
      ActivationDataSetter(learning_rate)));
  PadComponentFunctionWithInstruction(
      setup_size_init_, no_op_instruction, &algorithm.setup_);

  // Multiply with first layer weight matrix.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_MATRIX_PRODUCT_OP,
      kFirstLayerWeightsAddress, kFeaturesMatrixAddress,
      kFirstLayerOutputBeforeReluAddress));
  /// add matrix row mean op to generate a vector for the ease of later network.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_ROW_MEAN_OP,
      kFirstLayerOutputBeforeReluAddress,
      kFirstLayerOutputBeforeReluMatrixMeanAddress));
  // Add first layer bias.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP, kFirstLayerOutputBeforeReluMatrixMeanAddress, kFirstLayerBiasAddress,
      kFirstLayerOutputBeforeReluMatrixMeanAddress));
  // Apply RELU.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_MAX_OP, kFirstLayerOutputBeforeReluMatrixMeanAddress, kZerosAddress,
      kFirstLayerOutputAfterReluAddress));
  // Dot product with final layer weight vector.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP, kFirstLayerOutputAfterReluAddress,
      kFinalLayerWeightsAddress, kPredictionsScalarAddress));
  // Add final layer bias.
  CHECK_LE(kFinalLayerBiasAddress, kMaxScalarAddresses);
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP, kPredictionsScalarAddress, kFinalLayerBiasAddress,
      kPredictionsScalarAddress));
  PadComponentFunctionWithInstruction(
      predict_size_init_, no_op_instruction, &algorithm.predict_);

  algorithm.learn_.reserve(11);
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP, kLabelsScalarAddress, kPredictionsScalarAddress,
      kPredictionErrorAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      kLearningRateAddress, kPredictionErrorAddress, kPredictionErrorAddress));
  CHECK_LE(kFinalLayerBiasAddress, kMaxScalarAddresses);
  // Update final layer bias.
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
    SCALAR_SUM_OP, kFinalLayerBiasAddress, kPredictionErrorAddress,
    kFinalLayerBiasAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP, kPredictionErrorAddress,
      kFirstLayerOutputAfterReluAddress, kGradientWrtFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      kFinalLayerWeightsAddress, kGradientWrtFinalLayerWeightsAddress,
      kFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      kPredictionErrorAddress, kFinalLayerWeightsAddress,
      kGradientWrtActivationsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_HEAVYSIDE_OP,
      kFirstLayerOutputBeforeReluAddress, 0, kGradientOfReluAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_PRODUCT_OP,
      kGradientOfReluAddress, kGradientWrtActivationsAddress,
      kGradientWrtActivationsAddress));
  // Update first layer bias.
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
    VECTOR_SUM_OP, kFirstLayerBiasAddress, kGradientWrtActivationsAddress,
    kFirstLayerBiasAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_VECTOR_PRODUCT_OP,
      kGradientWrtActivationsAddress, kFeaturesMatrixAddress,
      kGradientWrtFirstLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_SUM_OP,
      kFirstLayerWeightsAddress, kGradientWrtFirstLayerWeightsAddress,
      kFirstLayerWeightsAddress));
  PadComponentFunctionWithInstruction(
      learn_size_init_, no_op_instruction, &algorithm.learn_);

  return algorithm;
}

Algorithm Generator::NeuralNet2(
    const double learning_rate,
    const double first_init_scale,
    const double final_init_scale) {
  Algorithm algorithm;

  // Scalar addresses
  constexpr AddressT kFinalLayerBiasAddress = 2;
  constexpr AddressT kLearningRateAddress = 3;
  constexpr AddressT kPredictionErrorAddress = 4;
  CHECK_GE(kMaxScalarAddresses, 5);

  // Vector addresses.
  constexpr AddressT kFirstLayerBiasAddress = 0;
  constexpr AddressT kFinalLayerWeightsAddress = 1;
  constexpr AddressT kFirstLayerOutputBeforeReluAddress= 2;
  constexpr AddressT kFirstLayerOutputAfterReluAddress = 3;
  constexpr AddressT kZerosAddress = 4;
  constexpr AddressT kGradientWrtFinalLayerWeightsAddress = 5;
  constexpr AddressT kGradientWrtActivationsAddress = 6;
  constexpr AddressT kGradientOfReluAddress = 7;
  constexpr AddressT kExtractRowAddress = 8;
  CHECK_GE(kMaxVectorAddresses, 9);

  // Matrix addresses.
  constexpr AddressT kFirstLayerWeightsAddress = 1;
  constexpr AddressT kGradientWrtFirstLayerWeightsAddress = 2;
  CHECK_GE(kMaxMatrixAddresses, 3);

  shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();

  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_GAUSSIAN_SET_OP,
      kFinalLayerWeightsAddress,
      FloatDataSetter(0.0),
      FloatDataSetter(final_init_scale)));
  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_GAUSSIAN_SET_OP,
      kFirstLayerWeightsAddress,
      FloatDataSetter(0.0),
      FloatDataSetter(first_init_scale)));
  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kLearningRateAddress,
      ActivationDataSetter(learning_rate)));
  PadComponentFunctionWithInstruction(
      setup_size_init_, no_op_instruction, &algorithm.setup_);

  // Get a row of feature matrix
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_ROW_OP,
      kExtractRowAddress,
      FloatDataSetter(0.85)));
  // Multiply with first layer weight matrix.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_VECTOR_PRODUCT_OP,
      kFirstLayerWeightsAddress, kExtractRowAddress,
      kFirstLayerOutputBeforeReluAddress));
  // Add first layer bias.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP, kFirstLayerOutputBeforeReluAddress, kFirstLayerBiasAddress,
      kFirstLayerOutputBeforeReluAddress));
  // Apply RELU.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_MAX_OP, kFirstLayerOutputBeforeReluAddress, kZerosAddress,
      kFirstLayerOutputAfterReluAddress));
  // Dot product with final layer weight vector.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP, kFirstLayerOutputAfterReluAddress,
      kFinalLayerWeightsAddress, kPredictionsScalarAddress));
  // Add final layer bias.
  CHECK_LE(kFinalLayerBiasAddress, kMaxScalarAddresses);
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP, kPredictionsScalarAddress, kFinalLayerBiasAddress,
      kPredictionsScalarAddress));
  PadComponentFunctionWithInstruction(
      predict_size_init_, no_op_instruction, &algorithm.predict_);

  algorithm.learn_.reserve(11);
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP, kLabelsScalarAddress, kPredictionsScalarAddress,
      kPredictionErrorAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      kLearningRateAddress, kPredictionErrorAddress, kPredictionErrorAddress));
  CHECK_LE(kFinalLayerBiasAddress, kMaxScalarAddresses);
  // Update final layer bias.
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
    SCALAR_SUM_OP, kFinalLayerBiasAddress, kPredictionErrorAddress,
    kFinalLayerBiasAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP, kPredictionErrorAddress,
      kFirstLayerOutputAfterReluAddress, kGradientWrtFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      kFinalLayerWeightsAddress, kGradientWrtFinalLayerWeightsAddress,
      kFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      kPredictionErrorAddress, kFinalLayerWeightsAddress,
      kGradientWrtActivationsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_HEAVYSIDE_OP,
      kFirstLayerOutputBeforeReluAddress, 0, kGradientOfReluAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_PRODUCT_OP,
      kGradientOfReluAddress, kGradientWrtActivationsAddress,
      kGradientWrtActivationsAddress));
  // Update first layer bias.
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
    VECTOR_SUM_OP, kFirstLayerBiasAddress, kGradientWrtActivationsAddress,
    kFirstLayerBiasAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_OUTER_PRODUCT_OP,
      kGradientWrtActivationsAddress, kExtractRowAddress,
      kGradientWrtFirstLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_SUM_OP,
      kFirstLayerWeightsAddress, kGradientWrtFirstLayerWeightsAddress,
      kFirstLayerWeightsAddress));
  PadComponentFunctionWithInstruction(
      learn_size_init_, no_op_instruction, &algorithm.learn_);

  return algorithm;
}

Algorithm Generator::NeuralNet3(
    const double learning_rate,
    const double first_init_scale,
    const double final_init_scale) {
  Algorithm algorithm;

  // Scalar addresses
  constexpr AddressT kFinalLayerBiasAddress = 2;
  constexpr AddressT kLearningRateAddress = 3;
  constexpr AddressT kPredictionErrorAddress = 4;
  constexpr AddressT kSoneAddress = 5;
  constexpr AddressT kStwoAddress = 6;
  constexpr AddressT kSthreeAddress = 7;
  constexpr AddressT kSfourAddress = 8;
  constexpr AddressT kSfiveAddress = 9;
  constexpr AddressT kSsixAddress = 10;
  constexpr AddressT kSsevenAddress = 11;
  constexpr AddressT kSeightAddress = 12;
  constexpr AddressT kSnineAddress = 13;
  CHECK_GE(kMaxScalarAddresses, 14);

  // Vector addresses.
  constexpr AddressT kFirstLayerBiasAddress = 0;
  constexpr AddressT kFinalLayerWeightsAddress = 1;
  constexpr AddressT kFirstLayerOutputBeforeReluAddress= 2;
  constexpr AddressT kFirstLayerOutputAfterReluAddress = 3;
  constexpr AddressT kZerosAddress = 4;
  constexpr AddressT kGradientWrtFinalLayerWeightsAddress = 5;
  constexpr AddressT kGradientWrtActivationsAddress = 6;
  constexpr AddressT kGradientOfReluAddress = 7;
  constexpr AddressT kExtractColumnAddress = 8;
  CHECK_GE(kMaxVectorAddresses, 9);

  // Matrix addresses.
  constexpr AddressT kFirstLayerWeightsAddress = 1;
  constexpr AddressT kGradientWrtFirstLayerWeightsAddress = 2;
  CHECK_GE(kMaxMatrixAddresses, 3);

  shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();

  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      VECTOR_GAUSSIAN_SET_OP,
      kFinalLayerWeightsAddress,
      FloatDataSetter(0.0),
      FloatDataSetter(final_init_scale)));
  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      MATRIX_GAUSSIAN_SET_OP,
      kFirstLayerWeightsAddress,
      FloatDataSetter(0.0),
      FloatDataSetter(first_init_scale)));
  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kLearningRateAddress,
      ActivationDataSetter(learning_rate)));
  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kStwoAddress,
      ActivationDataSetter(0.00001)));  
  PadComponentFunctionWithInstruction(
      setup_size_init_, no_op_instruction, &algorithm.setup_);

  algorithm.predict_.reserve(16);
  // james: leave some space for the first part alpha to evolve by adding some no_ops
  PadComponentFunctionWithInstruction(
      5, no_op_instruction, &algorithm.predict_);
  // Get a column of feature matrix
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_COLUMN_OP,
      kExtractColumnAddress,
      FloatDataSetter(0.99)));
  // james: leave some space for the first part alpha to evolve by adding some no_ops
  PadComponentFunctionWithInstruction(
      11, no_op_instruction, &algorithm.predict_);  
  // Multiply with first layer weight matrix.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_VECTOR_PRODUCT_OP,
      kFirstLayerWeightsAddress, kExtractColumnAddress,
      kFirstLayerOutputBeforeReluAddress));
  // Add first layer bias.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP, kFirstLayerOutputBeforeReluAddress, kFirstLayerBiasAddress,
      kFirstLayerOutputBeforeReluAddress));
  // Apply RELU.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_MAX_OP, kFirstLayerOutputBeforeReluAddress, kZerosAddress,
      kFirstLayerOutputAfterReluAddress));
  // Dot product with final layer weight vector.
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP, kFirstLayerOutputAfterReluAddress,
      kFinalLayerWeightsAddress, kPredictionsScalarAddress));
  // Add final layer bias.
  CHECK_LE(kFinalLayerBiasAddress, kMaxScalarAddresses);
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP, kPredictionsScalarAddress, kFinalLayerBiasAddress,
      kPredictionsScalarAddress));
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_SCALAR_OP, kSsixAddress, FloatDataSetter(0.852307),
      FloatDataSetter(0.999995))); 
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_SCALAR_OP, kSsevenAddress, FloatDataSetter(0.779995),
      FloatDataSetter(0.999995)));
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_SCALAR_OP, kSeightAddress, FloatDataSetter(0.702307),
      FloatDataSetter(0.999995))); 
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      MATRIX_GET_SCALAR_OP, kSnineAddress, FloatDataSetter(0.629995),
      FloatDataSetter(0.999995)));             
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP, kSsixAddress, kSnineAddress, 
      kSfiveAddress)); 
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP, kSeightAddress, 
      kSsevenAddress, kSfourAddress)); 
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP, kSfourAddress, 
      kStwoAddress, kSthreeAddress)); 
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIVISION_OP, kSfiveAddress, 
      kSthreeAddress, kSoneAddress)); 
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP, kStwoAddress, 
      kSoneAddress, kSoneAddress));
  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      SCALAR_SUM_OP, kPredictionsScalarAddress, 
      kSoneAddress, kPredictionsScalarAddress));                                   
  PadComponentFunctionWithInstruction(
      predict_size_init_, no_op_instruction, &algorithm.predict_);

  algorithm.learn_.reserve(11);
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP, kLabelsScalarAddress, kPredictionsScalarAddress,
      kPredictionErrorAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      kLearningRateAddress, kPredictionErrorAddress, kPredictionErrorAddress));
  CHECK_LE(kFinalLayerBiasAddress, kMaxScalarAddresses);
  // Update final layer bias.
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
    SCALAR_SUM_OP, kFinalLayerBiasAddress, kPredictionErrorAddress,
    kFinalLayerBiasAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP, kPredictionErrorAddress,
      kFirstLayerOutputAfterReluAddress, kGradientWrtFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      kFinalLayerWeightsAddress, kGradientWrtFinalLayerWeightsAddress,
      kFinalLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      kPredictionErrorAddress, kFinalLayerWeightsAddress,
      kGradientWrtActivationsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_HEAVYSIDE_OP,
      kFirstLayerOutputBeforeReluAddress, 0, kGradientOfReluAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_PRODUCT_OP,
      kGradientOfReluAddress, kGradientWrtActivationsAddress,
      kGradientWrtActivationsAddress));
  // Update first layer bias.
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
    VECTOR_SUM_OP, kFirstLayerBiasAddress, kGradientWrtActivationsAddress,
    kFirstLayerBiasAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_OUTER_PRODUCT_OP,
      kGradientWrtActivationsAddress, kExtractColumnAddress,
      kGradientWrtFirstLayerWeightsAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      MATRIX_SUM_OP,
      kFirstLayerWeightsAddress, kGradientWrtFirstLayerWeightsAddress,
      kFirstLayerWeightsAddress));
  PadComponentFunctionWithInstruction(
      learn_size_init_, no_op_instruction, &algorithm.learn_);

  return algorithm;
}

Algorithm Generator::LinearModel(const double learning_rate) {
  Algorithm algorithm;

  // Scalar addresses
  constexpr AddressT kLearningRateAddress = 2;
  constexpr AddressT kPredictionErrorAddress = 3;
  CHECK_GE(kMaxScalarAddresses, 4);

  // Vector addresses.
  constexpr AddressT kWeightsAddress = 1;
  constexpr AddressT kCorrectionAddress = 2;
  CHECK_GE(kMaxVectorAddresses, 3);

  CHECK_GE(kMaxMatrixAddresses, 0);

  shared_ptr<const Instruction> no_op_instruction =
      make_shared<const Instruction>();

  algorithm.setup_.emplace_back(make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kLearningRateAddress,
      ActivationDataSetter(learning_rate)));
  PadComponentFunctionWithInstruction(
      setup_size_init_, no_op_instruction, &algorithm.setup_);

  algorithm.predict_.emplace_back(make_shared<const Instruction>(
      VECTOR_INNER_PRODUCT_OP,
      kWeightsAddress, kFeaturesVectorAddress, kPredictionsScalarAddress));
  PadComponentFunctionWithInstruction(
      predict_size_init_, no_op_instruction, &algorithm.predict_);

  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_DIFF_OP,
      kLabelsScalarAddress, kPredictionsScalarAddress,
      kPredictionErrorAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_PRODUCT_OP,
      kLearningRateAddress, kPredictionErrorAddress,
      kPredictionErrorAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      SCALAR_VECTOR_PRODUCT_OP,
      kPredictionErrorAddress, kFeaturesVectorAddress, kCorrectionAddress));
  algorithm.learn_.emplace_back(make_shared<const Instruction>(
      VECTOR_SUM_OP,
      kWeightsAddress, kCorrectionAddress, kWeightsAddress));
  PadComponentFunctionWithInstruction(
      learn_size_init_, no_op_instruction, &algorithm.learn_);
  return algorithm;
}

}  // namespace alphaevolve
