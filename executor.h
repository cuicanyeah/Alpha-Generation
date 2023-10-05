// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Modifications copyright (C) 2021 <NUS/Cui Can>

#ifndef EXECUTOR_H_
#define EXECUTOR_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <fstream>

#include "task.pb.h"
#include "task.h"
#include "definitions.h"
#include "instruction.pb.h"
#include "algorithm.h"
#include "instruction.h"
#include "memory.h"
#include "random_generator.h"
#include "gtest/gtest_prod.h"

namespace alphaevolve {

using ::std::unique_ptr;  // NOLINT
using ::std::vector;  // NOLINT

namespace {
using ::std::cout;  // NOLINT
using ::std::endl;  // NOLINT
}  // namespace

constexpr FeatureIndexT kNumClasses = 13;
constexpr double kPadLabel = 0.0;

template <FeatureIndexT F>
Task<F>* MySafeDowncast(TaskInterface* task) {
  return dynamic_cast<Task<F>*>(task);
}
template <FeatureIndexT F>
const Task<F>* MySafeDowncast(const TaskInterface* task) {
  return dynamic_cast<const Task<F>*>(task);
}
template <FeatureIndexT F>
std::unique_ptr<Task<F>> MySafeDowncast(
    std::unique_ptr<TaskInterface>& task) {
  CHECK(task != nullptr);
  return std::unique_ptr<Task<F>>(MySafeDowncast<F>(task.release()));
}

template <FeatureIndexT F>
class Executor {
 public:
  // Constructs a standard executor. Uses a clean memory and automatically
  // executes the setup component function. All arguments are stored by
  // reference, so they must out-live the Executor instance.
  /// we don't add new operator here because new operator need iterator to loop over data samples.
  Executor(const Algorithm& algorithm, const Task<F>& dataset,
           // Includes the examples in all the training epochs.
           IntegerT num_all_train_examples, IntegerT num_valid_examples,
           RandomGenerator* rand_gen,
           // Errors larger than this trigger early stopping, as they signal
           // models that likely have runnaway behavior. Early stopping can also
           // be triggered if the loss for an example is infinite, nan, or too
           // large. If early stopping is triggered, the fitness for the
           // execution will be set to the minimum value.
           double max_abs_error);
  Executor(const Executor& other) = delete;
  Executor& operator=(const Executor& other) = delete;

  // Most code should use only the Execute method. Other methods below provide
  // lower-level access and can be used by tests and dataset generators. Returns
  // the fitness, according to the EvalType enum for the relevant dataset.
  double Execute(
      std::vector<double>* valid_preds = nullptr,
      std::vector<double>* price_diff = nullptr,
      std::vector<std::vector<std::vector<double>>>* tasks_rank = nullptr,
      IntegerT this_round = 0,
      IntegerT task_index = 0,
      IntegerT* num_stock_rank = nullptr, IntegerT* num_TS_rank = nullptr, const IntegerT num_of_stocks_to_approximate_rank = 99999,
      std::vector<double>* train_errors = nullptr,
      std::vector<double>* valid_errors = nullptr,
      std::vector<IntegerT>* useful_list = nullptr);
  // Get the number of train steps this executor has performed.
  IntegerT GetNumTrainStepsCompleted() const;

  // Use only from unit tests.
  inline Memory<F>& MemoryRef() {return memory_;}

 private:
  FRIEND_TEST(ExecutorTest, PredictComponentFunctionRuns);
  FRIEND_TEST(ExecutorTest, LearnComponentFunctionRuns);
  FRIEND_TEST(ExecutorTest, ItereatesThroughFeatures);
  FRIEND_TEST(ExecutorTest, ItereatesThroughLabelsDuringTraining);
  FRIEND_TEST(ExecutorTest, ValidationDoesNotSeeLabels);
  FRIEND_TEST(ExecutorTest, TrainOptimizationsAreCorrect);
  FRIEND_TEST(ExecutorTest, MultiEpochTrainingWorksCorrectly);

  // Performs training until the end. Returns whether successful. If not, it
  // means training stopped early.
  bool Train(std::vector<double>* errors);

  // Performs training for a given number of steps. Returns whether successful.
  // If not, it means training stopped early.
  bool Train(IntegerT max_steps, std::vector<double>* errors,
             // The iterators are used to track the training progress.
             // They should point to dataset_.train_features_.begin(),
             // dataset_.train_labels_.begin() and
             // dataset_.vector_train_labels_.begin() initially, and will be
             // updated after each training step.
             TaskIterator<F>* train_it, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_round, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list);

  // Check if features has -1234
  // template <FeatureIndexT F>
  bool CheckFeature(Matrix<F> features);

  // bool CheckHasIn(const Algorithm* algorithm, IntegerT ins_count, IntegerT in1);
  // bool CheckHasOut(const Algorithm* algorithm, IntegerT ins_count, IntegerT out, IntegerT check_out, IntegerT check_type);
  // Implementations of the train component function, with different
  // optimizations.
  bool TrainNoOptImpl(IntegerT max_steps, std::vector<double>* errors,
                      // See `Train` for more details about the following args.
                      TaskIterator<F>* train_it, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_round, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list);

  template <size_t max_component_function_size>
  bool TrainOptImpl(IntegerT max_steps, std::vector<double>* errors,
                    // See `Train` for more details about the following args.
                    TaskIterator<F>* train_it, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_round, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list);

  // Performs validation and returns the loss.
  double Validate(const IntegerT max_steps, std::vector<double>* errors, std::vector<double>* preds, std::vector<double>* price_diff, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_round, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list);

  // Copies memory_ into *memory. Useful for tests.
  void GetMemory(Memory<F>* memory);

  // The Algorithm being trained.
  const Algorithm& algorithm_;

  // The dataset used for training.
  const Task<F>& dataset_;

  const IntegerT num_all_train_examples_;
  const IntegerT num_valid_examples_;
  RandomGenerator* rand_gen_;
  Memory<F> memory_;
  Memory<F> mymemory_;
  Matrix<F> features;

  const double max_abs_error_;
  IntegerT num_train_steps_completed_;
};
static IntegerT previous_relation = -1;
static IntegerT current_relation = -1;
static IntegerT current_relation_count;
// Fills the training and validation labels, using the given Algorithm and
// memory. Can alter this memory, but only if the predict component function of
// the Algorithm does so--only runs the predict component function. Useful for
// dataset generators to generate labels.
template <FeatureIndexT F>
void ExecuteAndFillLabels(const Algorithm& algorithm, Memory<F>* memory,
                          TaskBuffer<F>* buffer,
                          RandomGenerator* rand_gen);

// Maps the interval [0.0, inf] to [0.0, 1.0]. The squashing is done by an
// arctan, so that losses in [0.0, 0.5] approximately undergo an affine
// transformation.
double FlipAndSquash(double value);

inline double Sigmoid(double x) {
  return static_cast<double>(1.0) /
      (static_cast<double>(1.0) + std::exp(-x));
}

namespace internal {

template<FeatureIndexT F>
inline Vector<F> TruncatingSoftmax(const Vector<F>& input);

template<FeatureIndexT F>
inline FeatureIndexT Argmax(const Vector<F>& input);

}  // namespace internal

////////////////////////////////////////////////////////////////////////////////
// Scalar arithmetic-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarSumOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] + memory->scalar_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarDiffOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] - memory->scalar_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] * memory->scalar_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarDivisionOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  if (memory->scalar_[instruction.in2_] == 0) {
      memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] /
      (memory->scalar_[instruction.in2_] + 0.001);
  } else {
      memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] /
      memory->scalar_[instruction.in2_];
  }
}

template<FeatureIndexT F>
inline void ExecuteScalarMinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::min(
      memory->scalar_[instruction.in1_],
      memory->scalar_[instruction.in2_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarMaxOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::max(
      memory->scalar_[instruction.in1_],
      memory->scalar_[instruction.in2_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarAbsOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::abs(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarHeavisideOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] >= 0.0 ? 1.0 : 0.0;
}

template<FeatureIndexT F>
inline void ExecuteScalarConstSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = instruction.GetActivationData();
}

template<FeatureIndexT F>
inline void ExecuteScalarReciprocalOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      static_cast<double>(1.0) / memory->scalar_[instruction.in1_];
}


////////////////////////////////////////////////////////////////////////////////
// Trigonometry-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarSinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::sin(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarCosOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::cos(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarTanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::tan(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarArcSinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::asin(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarArcCosOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::acos(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarArcTanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::atan(
      memory->scalar_[instruction.in1_]);
}


////////////////////////////////////////////////////////////////////////////////
// Calculus-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarExpOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::exp(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarLogOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] = std::log(
      memory->scalar_[instruction.in1_]);
}


////////////////////////////////////////////////////////////////////////////////
// Vector arithmetic-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteVectorSumOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in1_] + memory->vector_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteVectorDiffOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in1_] - memory->vector_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteVectorProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array() *
       memory->vector_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorDvisionOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array() /
       memory->vector_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorMinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array().min(
          memory->vector_[instruction.in2_].array())).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorMaxOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array().max(
          memory->vector_[instruction.in2_].array())).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorAbsOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in1_].array().abs().matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorHeavisideOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const double* in = memory->vector_[instruction.in1_].data();
  const double* in_end = in + F;
  double* out = memory->vector_[instruction.out_].data();
  while (in != in_end) {
    *out = *in > 0.0 ? 1.0 : 0.0;
    ++out;
    ++in;
  }
}

template<FeatureIndexT F>
inline void ExecuteVectorConstSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const FeatureIndexT index =
      FloatToIndex(instruction.GetFloatData0(), F);
  memory->vector_[instruction.out_](index) = instruction.GetFloatData1();
}

template<FeatureIndexT F>
inline void ExecuteVectorReciprocalOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      (static_cast<double>(1.0) /
       memory->vector_[instruction.in1_].array())
          .matrix();
}


////////////////////////////////////////////////////////////////////////////////
// Matrix arithmetic-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteMatrixSumOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_] + memory->matrix_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteMatrixDiffOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_] - memory->matrix_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteMatrixProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      (memory->matrix_[instruction.in1_].array() *
       memory->matrix_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteMatrixDivisionOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      (memory->matrix_[instruction.in1_].array() /
       memory->matrix_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteMatrixMinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const double* in1 = memory->matrix_[instruction.in1_].data();
  const double* in2 = memory->matrix_[instruction.in2_].data();
  const double* in1_end = in1 + F * F;
  double* out = memory->matrix_[instruction.out_].data();
  while (in1 != in1_end) {
    const double in1v = *in1;
    const double in2v = *in2;
    *out = in1v < in2v ? in1v : in2v;
    ++out;
    ++in1;
    ++in2;
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixMaxOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const double* in1 = memory->matrix_[instruction.in1_].data();
  const double* in2 = memory->matrix_[instruction.in2_].data();
  const double* in1_end = in1 + F * F;
  double* out = memory->matrix_[instruction.out_].data();
  while (in1 != in1_end) {
    const double in1v = *in1;
    const double in2v = *in2;
    *out = in1v > in2v ? in1v : in2v;
    ++out;
    ++in1;
    ++in2;
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixAbsOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_].array().abs().matrix();
}

template<FeatureIndexT F>
inline void ExecuteMatrixHeavisideOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const double* in = memory->matrix_[instruction.in1_].data();
  const double* in_end = in + F * F;
  double* out = memory->matrix_[instruction.out_].data();
  while (in != in_end) {
    *out = *in > 0.0 ? 1.0 : 0.0;
    ++out;
    ++in;
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixConstSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_](
      FloatToIndex(instruction.GetFloatData0(), F),
      FloatToIndex(instruction.GetFloatData1(), F)) =
          instruction.GetFloatData2();
}

template<FeatureIndexT F>
inline void ExecuteMatrixReciprocalOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      (static_cast<double>(1.0) /
       memory->matrix_[instruction.in1_].array())
          .matrix();
}


////////////////////////////////////////////////////////////////////////////
// Linear algebra-related instructions.
////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarVectorProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in2_] * memory->scalar_[instruction.in1_];
}

template<FeatureIndexT F>
inline void ExecuteVectorInnerProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->vector_[instruction.in1_].dot(
          memory->vector_[instruction.in2_]);
}

template<FeatureIndexT F>
inline void ExecuteVectorOuterProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      memory->vector_[instruction.in1_] *
          memory->vector_[instruction.in2_].transpose();
}

template<FeatureIndexT F>
inline void ExecuteScalarMatrixProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in2_] * memory->scalar_[instruction.in1_];
}

template<FeatureIndexT F>
inline void ExecuteMatrixVectorProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_] * memory->vector_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteVectorNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->vector_[instruction.in1_].norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->matrix_[instruction.in1_].norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixRowNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_].rowwise().norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixColumnNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_].colwise().norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixTransposeOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  if (instruction.out_ == instruction.in1_) {
    memory->matrix_[instruction.in1_].transposeInPlace();
  } else {
    memory->matrix_[instruction.out_] =
        memory->matrix_[instruction.in1_].transpose();
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixMatrixProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> feature) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_] * memory->matrix_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarBroadcastOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->scalar_[instruction.in1_] * Vector<F>::Ones(F);
}

template<FeatureIndexT F>
inline void ExecuteVectorColumnBroadcastOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      memory->vector_[instruction.in1_].replicate(1, F);
}

template<FeatureIndexT F>
inline void ExecuteVectorRowBroadcastOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->matrix_[instruction.out_] =
      memory->vector_[instruction.in1_].replicate(1, F).transpose();
}


////////////////////////////////////////////////////////////////////////////////
// Probability-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteVectorMeanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->vector_[instruction.in1_].mean();
}

template<FeatureIndexT F>
inline void ExecuteVectorStDevOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const Vector<F>& values = memory->vector_[instruction.in1_];
  const double mean = values.mean();
  memory->scalar_[instruction.out_] =
      sqrt(values.dot(values) / static_cast<double>(F) -
           mean * mean);
}

template<FeatureIndexT F>
inline void ExecuteMatrixMeanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      memory->matrix_[instruction.in1_].mean();
}

template<FeatureIndexT F>
inline void ExecuteMatrixStDevOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const Matrix<F>& values = memory->matrix_[instruction.in1_];
  const double mean = values.mean();
  memory->scalar_[instruction.out_] =
      sqrt((values.array() * values.array()).sum() /
           static_cast<double>(F * F) -
           mean * mean);
}

template<FeatureIndexT F>
inline void ExecuteMatrixRowMeanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_].rowwise().mean();
}

template<FeatureIndexT F>
inline void ExecuteMatrixRowStDevOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  for (IntegerT row = 0; row < F; ++row) {
    const Vector<F>& values =
        memory->matrix_[instruction.in1_].row(row);
    const double mean = values.mean();
    const double stdev =
        sqrt((values.array() * values.array()).sum() /
             static_cast<double>(F) -
             mean * mean);
    memory->vector_[instruction.out_](row) = stdev;
  }
}

template<FeatureIndexT F>
inline void ExecuteScalarGaussianSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      rand_gen->GaussianActivation(
          instruction.GetFloatData0(), instruction.GetFloatData1());
}

template<FeatureIndexT F>
inline void ExecuteVectorGaussianSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  rand_gen->FillGaussian<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->vector_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteMatrixGaussianSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  rand_gen->FillGaussian<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->matrix_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarUniformSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
      rand_gen->UniformActivation(
          instruction.GetFloatData0(), instruction.GetFloatData1());
}

template<FeatureIndexT F>
inline void ExecuteVectorUniformSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  rand_gen->FillUniform<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->vector_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteMatrixUniformSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  rand_gen->FillUniform<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->matrix_[instruction.out_]);
}


////////////////////////////////////////////////////////////////////////////////
// My instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteGetScalarOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features_it) {
    double features_element = features_it(
      FloatToIndex(instruction.GetFloatData0(), F),
      FloatToIndex(instruction.GetFloatData1(), F));

    memory->scalar_[instruction.out_] = features_element;
}

template<FeatureIndexT F>
inline void ExecutePreviousScalarRankOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
  Memory<F>* memory, const Matrix<F> features) {
}

template<FeatureIndexT F>
inline void ExecuteRelationScalarRankOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
  Memory<F>* memory, const Matrix<F> features) {
}

template<FeatureIndexT F>
inline void ExecuteRelationScalarDemeanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
  Memory<F>* memory, const Matrix<F> features) {
}

template<FeatureIndexT F>
inline void ExecuteTSRowRankOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
  Memory<F>* memory, const Matrix<F> features) {
}

template<FeatureIndexT F>
inline void ExecuteTSRankOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
  Memory<F>* memory, const Matrix<F> features) {
}

template<FeatureIndexT F>
inline void ExecuteConditionOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->scalar_[instruction.out_] =
    (memory->scalar_[instruction.in1_] > memory->scalar_[instruction.in2_]) ? memory->scalar_[instruction.in3_] : memory->scalar_[instruction.in4_];
}

template<FeatureIndexT F>
inline void ExecuteGetColumnOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] = features.col(FloatToIndex(instruction.GetFloatData0(), F));
}

template<FeatureIndexT F>
inline void ExecuteGetRowOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  memory->vector_[instruction.out_] = features.row(FloatToIndex(instruction.GetFloatData0(), F));
}

template<FeatureIndexT F>
inline void ExecuteCorrelationOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  const Vector<F>& valuesA = memory->vector_[instruction.in1_];
  const Vector<F>& valuesB = memory->vector_[instruction.in2_];

  const double meanA = valuesA.mean();
  const double stdevA =
      sqrt((valuesA.array() * valuesA.array()).sum() /
           static_cast<double>(F) -
           meanA * meanA);
  const double meanB = valuesB.mean();
  const double stdevB =
      sqrt((valuesB.array() * valuesB.array()).sum() /
           static_cast<double>(F) -
           meanB * meanB);    
  if (isnan(stdevA) || isnan(stdevB) || stdevA == 0 || stdevB == 0 || meanA == 0 || meanB == 0 || (valuesA.size() - 1) == 0) {
    memory->scalar_[instruction.out_] = 0;
  } else {    
    memory->scalar_[instruction.out_] = ((((valuesA).array() - meanA) / stdevA * ((valuesB).array() - meanB) / stdevB).sum())/ (valuesA.size() - 1);
  }
  if (isnan(memory->scalar_[instruction.out_])) {
    CHECK_EQ(stdevA, 0);
    CHECK(!isnan(memory->scalar_[instruction.out_]));
  }
}

template<FeatureIndexT F>
inline void ExecuteCovarianceOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {

  const Vector<F>& valuesA = memory->vector_[instruction.in1_];
  const Vector<F>& valuesB = memory->vector_[instruction.in2_];

  const double meanA = valuesA.mean();
  const double stdevA =
      sqrt((valuesA.array() * valuesA.array()).sum() /
           static_cast<double>(F) -
           meanA * meanA);
  const double meanB = valuesB.mean();
  const double stdevB =
      sqrt((valuesB.array() * valuesB.array()).sum() /
           static_cast<double>(F) -
           meanB * meanB);    

  if (isnan(stdevA) || isnan(stdevB) || stdevA == 0 || stdevB == 0 || meanA == 0 || meanB == 0 || (valuesA.size() - 1) == 0) {
    memory->scalar_[instruction.out_] = 0;
  } else {    
    memory->scalar_[instruction.out_] = ((((valuesA).array() - meanA) * ((valuesB).array() - meanB)).sum())/ (valuesA.size() - 1);
  }
  if (isnan(memory->scalar_[instruction.out_])) {
    for (IntegerT for_compare = 0; for_compare < valuesA.size(); ++for_compare) {
      std::cout << "debug valuesA(for_compare) " << valuesA(for_compare) << std::endl;
      std::cout << "debug valuesB(for_compare)? " << valuesB(for_compare) << std::endl;
    }
    CHECK(!isnan(memory->scalar_[instruction.out_]));
  }
}

////////////////////////////////////////////////////////////////////////////////
// Other instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteNoOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
}

template<FeatureIndexT F>
inline void ExecuteUnsupportedOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
  LOG(FATAL) << "Unsupported op." << std::endl;
}

template<FeatureIndexT F>
static constexpr std::array<
    void(*)(const Instruction&, RandomGenerator*, Memory<F>*, const Matrix<F>),
    128> kOpIndexToExecuteFunction = {
        &ExecuteNoOp<F>,                   // NO_OP = 0
        &ExecuteScalarSumOp<F>,            // SCALAR_SUM_OP = 1
        &ExecuteScalarDiffOp,              // SCALAR_DIFF_OP = 2
        &ExecuteScalarProductOp,           // SCALAR_PRODUCT_OP = 3
        &ExecuteScalarDivisionOp,          // SCALAR_DIVISION_OP = 4
        &ExecuteScalarAbsOp,               // SCALAR_ABS_OP = 5
        &ExecuteScalarReciprocalOp,        // SCALAR_RECIPROCAL_OP = 6
        &ExecuteScalarSinOp,               // SCALAR_SIN_OP = 7
        &ExecuteScalarCosOp,               // SCALAR_COS_OP = 8
        &ExecuteScalarTanOp,               // SCALAR_TAN_OP = 9
        &ExecuteScalarArcSinOp,            // SCALAR_ARCSIN_OP = 10
        &ExecuteScalarArcCosOp,            // SCALAR_ARCCOS_OP = 11
        &ExecuteScalarArcTanOp,            // SCALAR_ARCTAN_OP = 12
        &ExecuteScalarExpOp,               // SCALAR_EXP_OP = 13
        &ExecuteScalarLogOp,               // SCALAR_LOG_OP = 14
        &ExecuteScalarHeavisideOp,         // SCALAR_HEAVYSIDE_OP = 15
        &ExecuteVectorHeavisideOp,         // VECTOR_HEAVYSIDE_OP = 16
        &ExecuteMatrixHeavisideOp,         // MATRIX_HEAVYSIDE_OP = 17
        &ExecuteScalarVectorProductOp,     // SCALAR_VECTOR_PRODUCT_OP = 18
        &ExecuteScalarBroadcastOp,         // SCALAR_BROADCAST_OP = 19
        &ExecuteVectorReciprocalOp,        // VECTOR_RECIPROCAL_OP = 20
        &ExecuteVectorNormOp,              // VECTOR_NORM_OP = 21
        &ExecuteVectorAbsOp,               // VECTOR_ABS_OP = 22
        &ExecuteVectorSumOp,               // VECTOR_SUM_OP = 23
        &ExecuteVectorDiffOp,              // VECTOR_DIFF_OP = 24
        &ExecuteVectorProductOp,           // VECTOR_PRODUCT_OP = 25
        &ExecuteVectorDvisionOp,           // VECTOR_DIVISION_OP = 26
        &ExecuteVectorInnerProductOp,      // VECTOR_INNER_PRODUCT_OP = 27
        &ExecuteVectorOuterProductOp,      // VECTOR_OUTER_PRODUCT_OP = 28
        &ExecuteScalarMatrixProductOp,     // SCALAR_MATRIX_PRODUCT_OP = 29
        &ExecuteMatrixReciprocalOp,        // MATRIX_RECIPROCAL_OP = 30
        &ExecuteMatrixVectorProductOp<F>,  // MATRIX_VECTOR_PRODUCT_OP = 31
        &ExecuteVectorColumnBroadcastOp,   // VECTOR_COLUMN_BROADCAST_OP = 32
        &ExecuteVectorRowBroadcastOp,      // VECTOR_ROW_BROADCAST_OP = 33
        &ExecuteMatrixNormOp,              // MATRIX_NORM_OP = 34
        &ExecuteMatrixColumnNormOp,        // MATRIX_COLUMN_NORM_OP = 35
        &ExecuteMatrixRowNormOp,           // MATRIX_ROW_NORM_OP = 36
        &ExecuteMatrixTransposeOp,         // MATRIX_TRANSPOSE_OP = 37
        &ExecuteMatrixAbsOp,               // MATRIX_ABS_OP = 38
        &ExecuteMatrixSumOp,               // MATRIX_SUM_OP = 39
        &ExecuteMatrixDiffOp,              // MATRIX_DIFF_OP = 40
        &ExecuteMatrixProductOp,           // MATRIX_PRODUCT_OP = 41
        &ExecuteMatrixDivisionOp,          // MATRIX_DIVISION_OP = 42
        &ExecuteMatrixMatrixProductOp,     // MATRIX_MATRIX_PRODUCT_OP = 43
        &ExecuteScalarMinOp,               // SCALAR_MIN_OP = 44
        &ExecuteVectorMinOp,               // VECTOR_MIN_OP = 45
        &ExecuteMatrixMinOp,               // MATRIX_MIN_OP = 46
        &ExecuteScalarMaxOp,               // SCALAR_MAX_OP = 47
        &ExecuteVectorMaxOp,               // VECTOR_MAX_OP = 48
        &ExecuteMatrixMaxOp,               // MATRIX_MAX_OP = 49
        &ExecuteVectorMeanOp<F>,           // VECTOR_MEAN_OP = 50
        &ExecuteMatrixMeanOp,              // MATRIX_MEAN_OP = 51
        &ExecuteMatrixRowMeanOp,           // MATRIX_ROW_MEAN_OP = 52
        &ExecuteMatrixRowStDevOp,          // MATRIX_ROW_ST_DEV_OP = 53
        &ExecuteVectorStDevOp,             // VECTOR_ST_DEV_OP = 54
        &ExecuteMatrixStDevOp,             // MATRIX_ST_DEV_OP = 55
        &ExecuteScalarConstSetOp,          // SCALAR_CONST_SET_OP = 56
        &ExecuteVectorConstSetOp,          // VECTOR_CONST_SET_OP = 57
        &ExecuteMatrixConstSetOp,          // MATRIX_CONST_SET_OP = 58
        &ExecuteScalarUniformSetOp,        // SCALAR_UNIFORM_SET_OP = 59
        &ExecuteVectorUniformSetOp,        // VECTOR_UNIFORM_SET_OP = 60
        &ExecuteMatrixUniformSetOp,        // MATRIX_UNIFORM_SET_OP = 61
        &ExecuteScalarGaussianSetOp,       // SCALAR_GAUSSIAN_SET_OP = 62
        &ExecuteVectorGaussianSetOp,       // VECTOR_GAUSSIAN_SET_OP = 63
        &ExecuteMatrixGaussianSetOp,       // MATRIX_GAUSSIAN_SET_OP = 64
        &ExecuteRelationScalarRankOp,      // MATRIX_RELATION_RANK_OP = 65
        &ExecutePreviousScalarRankOp,      // MATRIX_PREVIOUS_RANK_OP = 66
        &ExecuteGetScalarOp,               // MATRIX_GET_SCALAR_OP = 67
        &ExecuteGetColumnOp,               // MATRIX_GET_COLUMN_OP = 68
        &ExecuteGetRowOp,                  // MATRIX_GET_ROW_OP = 69
        &ExecuteConditionOp,               // CONDITION_OP = 70
        &ExecuteCorrelationOp,             // CORRELATION_OP = 71
        &ExecuteTSRankOp,                  // TSRANK_OP = 72
        &ExecuteTSRowRankOp,               // TS_ROW_RANK_OP = 73
        &ExecuteCovarianceOp,              // COVARIANCE_OP = 74
        &ExecuteRelationScalarDemeanOp,    // RELATION_DEMEAN_OP = 75
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 76
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 77
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 78
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 79
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 80
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 81
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 82
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 83
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 84
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 85
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 86
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 87
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 88
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 89
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 90
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 91
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 92
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 93
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 94
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 95
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 96
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 97
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 98
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 99
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 100
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 101
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 102
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 103
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 104
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 105
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 106
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 107
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 108
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 109
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 110
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 111
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 112
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 113
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 114
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 115
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 116
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 117
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 118
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 119
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 120
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 121
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 122
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 123
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 124
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 125
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 126
        &ExecuteUnsupportedOp              // UNSUPPORTED_OP = 127
    };

template<FeatureIndexT F>
inline void ExecuteInstruction(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const Matrix<F> dummy_features; 
  (*kOpIndexToExecuteFunction<F>[instruction.op_])(
      instruction, rand_gen, memory, dummy_features); 
}

template<FeatureIndexT F>
inline void ExecuteMyInstruction(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory, const Matrix<F> features) {
    (*kOpIndexToExecuteFunction<F>[instruction.op_])(
        instruction, rand_gen, memory, features);
}

template <FeatureIndexT F>
struct ZeroLabelAssigner {
  inline static void Assign(Memory<F>* memory) {
    memory->scalar_[kLabelsScalarAddress] = 0.0;
  }
};

template <FeatureIndexT F>
struct LabelAssigner {
  inline static void Assign(const Scalar& label, Memory<F>* memory) {
    memory->scalar_[kLabelsScalarAddress] = label;
  }
};

template<FeatureIndexT F>
struct PredictionGetter {
  inline static Scalar Get(Memory<F>* memory) {
    return memory->scalar_[kPredictionsScalarAddress];
  }
};

template <FeatureIndexT F>
struct ErrorComputer {
  inline static double Compute(const Memory<F>& memory, const Scalar& label) {
    return std::abs(label - memory.scalar_[kPredictionsScalarAddress]);
  }
};

template <FeatureIndexT F>
struct ProbabilityConverter {
  inline static void Convert(Memory<F>* memory) {
    memory->scalar_[kPredictionsScalarAddress] =
        2 * Sigmoid(memory->scalar_[kPredictionsScalarAddress]) - 1; // because return ranges from 1 to -1
  }
};

template <FeatureIndexT F>
Executor<F>::Executor(const Algorithm& algorithm, const Task<F>& dataset,
                      const IntegerT num_all_train_examples,
                      const IntegerT num_valid_examples,
                      RandomGenerator* rand_gen,
                      const double max_abs_error)
    : algorithm_(algorithm),
      dataset_(dataset),
      num_all_train_examples_(num_all_train_examples),
      num_valid_examples_(num_valid_examples),
      rand_gen_(rand_gen),
      max_abs_error_(max_abs_error),
      num_train_steps_completed_(0) {
  memory_.Wipe();
  mymemory_.Wipe();
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm_.setup_) {
    ExecuteInstruction(*instruction, rand_gen_, &memory_);
    ExecuteInstruction(*instruction, rand_gen_, &mymemory_);
  }
}

template <FeatureIndexT F>
double Executor<F>::Execute(std::vector<double>* valid_preds,
                            std::vector<double>* price_diff,
                            std::vector<std::vector<std::vector<double>>>* tasks_rank,
                            IntegerT this_round,
                            IntegerT task_index,
                            IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<double>* train_errors,
                            std::vector<double>* valid_errors, std::vector<IntegerT> *useful_list) {
  CHECK_GE(dataset_.NumTrainEpochs(), 1);

  // Iterators that track the progresss of training.
  TaskIterator<F> train_it = dataset_.TrainIterator();

  // Train for multiple epochs, evaluate on validation set
  // after each epoch and take the best validation result as fitness.
  const IntegerT num_all_train_examples =
      std::min(num_all_train_examples_,
               static_cast<IntegerT>(dataset_.MaxTrainExamples()));
  const IntegerT num_examples_per_epoch =
      dataset_.TrainExamplesPerEpoch() == kNumTrainExamplesNotSet ?
      num_all_train_examples : dataset_.TrainExamplesPerEpoch();
  IntegerT num_remaining = num_all_train_examples;
  double best_fitness = kMinFitness;
  while (num_remaining > 0) {
    if (!Train(
            std::min(num_examples_per_epoch, num_remaining),
            train_errors, &train_it, tasks_rank, this_round, 
            task_index, num_stock_rank, num_TS_rank, 
            num_of_stocks_to_approximate_rank, useful_list)) {
      if (num_remaining == num_all_train_examples) {
        return kMinFitness;
      } else {
        break;
      }
    } 
    num_remaining -= num_examples_per_epoch;
    if (valid_preds != nullptr && price_diff != nullptr) {
      valid_preds->clear();
      price_diff->clear();
    }
    const double current_fitness = Validate(num_examples_per_epoch, valid_errors, valid_preds, price_diff, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, useful_list);
    // std::string filename_returns = "/hdd7/james/HSI_AlphaEvolve/debug_can_delete_why_dont_match/long_data/" +std::to_string(task_index)+ "_preds.txt";  
    // std::ofstream outFilepreds(filename_returns);
    // for (auto i = valid_preds->begin(); i != valid_preds->end(); ++i) {
    //   outFilepreds << (*i) << " ";
    // }      
    // outFilepreds.close(); 
    // std::string filename_price_diff = "/hdd7/james/HSI_AlphaEvolve/debug_can_delete_why_dont_match/long_data/" +std::to_string(task_index)+ "_labels.txt";  
    // std::ofstream outFilelabels(filename_price_diff);
    // for (auto i = price_diff->begin(); i != price_diff->end(); ++i) {
    //   outFilelabels << (*i) << " ";
    // }      
    // outFilelabels.close();   
    best_fitness = std::max(current_fitness, best_fitness);
    // Only save the errors of the first epoch.
    if (train_errors != nullptr) {
      train_errors = nullptr;
      valid_errors = nullptr;
    }
  }

  return best_fitness;
}

template <FeatureIndexT F>
IntegerT Executor<F>::GetNumTrainStepsCompleted() const {
  return num_train_steps_completed_;
}

template <FeatureIndexT F>
bool Executor<F>::Train(std::vector<double>* errors) {
  // Iterators that tracks the progresss of training.
  typename std::vector<Matrix<F>>::const_iterator train_feature_it =
      dataset_.train_features_.begin();
  cout << " dataset_.train_features_.begin() " << dataset_.train_features_.begin() << endl;
  typename std::vector<Scalar>::const_iterator train_label_it =
      dataset_.train_labels_.begin();
  const IntegerT num_all_train_examples =
      std::min(num_all_train_examples_,
               static_cast<IntegerT>(dataset_.train_features_.size()));
  return Train(num_all_train_examples, errors, &train_feature_it,
               &train_label_it);
}

// At or above these many steps, we optimize the train component function.
constexpr IntegerT kTrainStepsOptThreshold = 1000;

template <FeatureIndexT F>
bool Executor<F>::Train(const IntegerT max_steps, std::vector<double>* errors,
                        TaskIterator<F>* train_it, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_round, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list) {
  CHECK(errors == nullptr || max_steps <= 100) <<
      "You should only record the training errors for few training steps."
      << std::endl;
  if (max_steps < kTrainStepsOptThreshold) {
    return TrainNoOptImpl(max_steps, errors, train_it, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, useful_list);
  } else {
    if (algorithm_.predict_.size() <= 10 && algorithm_.learn_.size() <= 10) {
      return TrainOptImpl<10>(max_steps, errors, train_it, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, useful_list);
    } else if (algorithm_.predict_.size() <= 100 &&
               algorithm_.learn_.size() <= 100) {
      return TrainOptImpl<100>(max_steps, errors, train_it, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, useful_list);
    } else if (algorithm_.predict_.size() <= 1000 &&
               algorithm_.learn_.size() <= 1000) {
      return TrainOptImpl<1000>(max_steps, errors, train_it, tasks_rank, this_round, task_index, num_stock_rank, num_TS_rank, num_of_stocks_to_approximate_rank, useful_list);
    } else {
      LOG(FATAL) << "ComponentFunction size not yet supported." << std::endl;
    }
  }
}


template <FeatureIndexT F>
bool Executor<F>::CheckFeature(Matrix<F> features) {
      for (IntegerT i = 0; i < F; ++i) {
        for (IntegerT j = 0; j < F; ++j) {
          if (std::abs((features(i ,j))) == 1234) return true;
      } 
    }
    return false;
}

// We don't care that this function is inline. We just want to keep it here,
// next to TrainOptImpl (below).
template <FeatureIndexT F>
inline bool Executor<F>::TrainNoOptImpl(const IntegerT max_steps,
                                        std::vector<double>* errors,
                                        TaskIterator<F>* train_it,
                                        std::vector<std::vector<std::vector<double>>>* tasks_rank,
                                        IntegerT this_round,
                                        IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list) {
  if (errors != nullptr) {
    errors->reserve(max_steps);
  }
  vector<double> relatiion_start_list = {0,36,69,74,79,235,237,369,386,387,406,411,452,464,508,524,530,539,543,551,553,570,599,605,616,622,635,643,661,663,670,671,691,700,701,707,718,721,724,726,730,732,734,744,749,757,764,767,775,782,802,805,809,813,816,822,825,830,833,837,843,850,852,860,862,866,869,875,880,882,890,891,896,898,903,904,908,910,915,921,928,940,946,952,954,957,964,965,966,971,975,979,984,992,994,995,997,1000,1003,1004,1005,1007,1008,1009,1010,1012,1014,1018,1020,1021,1022,1024,1025};

  if (dataset_.industry_relation_ == current_relation) {
    ++current_relation_count;
    previous_relation = current_relation; // comparing the real current (dataset_.industry_relation_) with the actual previous (current_relation)
  } else current_relation_count = 0;
  current_relation = dataset_.industry_relation_;  
  
  

  for (IntegerT step = 0; step < max_steps; ++step) {
    num_train_steps_completed_++;
    // Run predict component function for this example.
    const Matrix<F>& features = train_it->GetFeatures();
    if (CheckFeature(features)) {
      train_it->Next();
      if (train_it->Done()) {
        break;  // Reached the end of the dataset.
      }
      continue;
    }
    memory_.matrix_[kFeaturesMatrixAddress] = features;
    ZeroLabelAssigner<F>::Assign(&memory_);

    // count how many instructions have passed
    IntegerT ins_count = 0;

    // count the num of rank_op we are dealing now
    IntegerT num_rank_op = 0;

    *num_stock_rank = 0;
    *num_TS_rank = 0;

    IntegerT num_stock_rank_count = 0;
    IntegerT num_TS_rank_count = 0;

    /// don't learn this sample because of rank op not enough data to compare yet
    bool if_skip_this_sample = false; 

    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm_.predict_) {

      double out = 0;

      if ((*useful_list)[algorithm_.learn_.size() + ins_count] < 1) {
        ++ins_count;
        continue;
      }
      if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 73 || instruction->op_ == 75) {

        if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 75) {
          if(isnan(memory_.scalar_[instruction->in1_])) {
            ++ins_count;
            continue;
          }
          if (instruction->op_ == 66 || instruction->op_ == 75 || instruction->op_ == 65) ++num_stock_rank_count;
          if (instruction->op_ == 72) ++num_TS_rank_count;
        } else if (instruction->op_ == 73) {
          if(isnan(memory_.scalar_[instruction->in1_])) {
              ++ins_count;
              continue;
            }
          ++num_stock_rank_count;
          ++num_TS_rank_count;         
        }
 
        switch (instruction->op_) {
          case 75: {
            if (((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0) { 
              IntegerT relation_len;
              if (dataset_.industry_relation_ != 112) relation_len = relatiion_start_list[dataset_.industry_relation_ + 1] - relatiion_start_list[dataset_.industry_relation_];
              else relation_len = 1;
              if (relation_len > 5) {
                for (IntegerT for_compare = relatiion_start_list[dataset_.industry_relation_]; for_compare < relatiion_start_list[dataset_.industry_relation_ + 1]; ++for_compare) {
                  out+=(*tasks_rank)[num_rank_op][for_compare][step];
                }                
              } else memory_.scalar_[instruction->out_] = 0;

              memory_.scalar_[instruction->out_] = memory_.scalar_[instruction->in1_] - out/relation_len;
            }

            if ((this_round + 1) == num_stock_rank_count) {
              (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
            }
            if(!(((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0)) if_skip_this_sample = true;

            break;  
          }
          case 65: {
            if (((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0) {
              IntegerT relation_len;
              if (dataset_.industry_relation_ != 112) relation_len = relatiion_start_list[dataset_.industry_relation_ + 1] - relatiion_start_list[dataset_.industry_relation_];
              else relation_len = 1;
              if (relation_len > 5) {
                for (IntegerT for_compare = relatiion_start_list[dataset_.industry_relation_]; for_compare < relatiion_start_list[dataset_.industry_relation_ + 1]; ++for_compare) {
                  if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][for_compare][step]) {
                    ++out;
                  }
                }

                memory_.scalar_[instruction->out_] = out/relation_len;
              } else memory_.scalar_[instruction->out_] = 0;
            }
            // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock
            if ((this_round + 1) == num_stock_rank_count) {
              (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
            }
            if(!((this_round + 1) > num_stock_rank_count && num_stock_rank_count > 0)) if_skip_this_sample = true;

            break;  
          }
          case 66: {
            ++(*num_stock_rank);
            
            if (this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) {   
              for (IntegerT for_compare = 0; for_compare < num_of_stocks_to_approximate_rank; ++for_compare) {
                if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][for_compare][step]) {
                  ++out;
                }

              }
              memory_.scalar_[instruction->out_] = out/(num_of_stocks_to_approximate_rank);
              
            }
            // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock
            if (this_round + 1 == num_stock_rank_count) (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
            if(!(this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0)) if_skip_this_sample = true;

            break;  
          }
          case 73: { 
            if ((this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) && step > (13 * (*num_TS_rank) + 12)) { /// if step is larger then 12, means there are 13 steps before you can compare
              IntegerT count = 0;
              for (IntegerT for_compare_step = step - 1 ; for_compare_step > (step - 2 - 12); --for_compare_step) { /// limit case is 0 to 12
                double count_out = 0.0;
                for (IntegerT for_compare = 0; for_compare < num_of_stocks_to_approximate_rank; ++for_compare) {
                  if ((*tasks_rank)[num_rank_op][task_index][for_compare_step] > (*tasks_rank)[num_rank_op][for_compare][for_compare_step]) ++count_out;
                }
                memory_.vector_[instruction->out_](count) = count_out/num_of_stocks_to_approximate_rank;
                ++count;
              }
            }
            // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock e.g. -1 < task_index < 10 or 9 < task_index < 20 or  19 < task_index < 30 ... 
            if ((this_round + 1 == num_stock_rank_count) && (13 * ((*num_TS_rank) - 1) + 12) < step) {
              (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
            }
            if (!((this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) && step > (13 * (*num_TS_rank) + 12))) if_skip_this_sample = true;

            ++(*num_stock_rank);
            ++(*num_TS_rank);        

            break;
          }
          case 72: {

            if (step > (13 * (*num_TS_rank) + 12)) {
              for (IntegerT for_compare = step - 1; for_compare > (step - 2 - 12); --for_compare) {
                if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][task_index][for_compare]) ++out;
                }
                memory_.scalar_[instruction->out_] = out/13;    
              }
            if ((13 * ((*num_TS_rank) - 1) + 12) < step) {
              (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
            }
            if (!(step > (13 * (*num_TS_rank) + 12))) if_skip_this_sample = true;

            ++(*num_TS_rank);        

            break;
          }  
          default:
            memory_.scalar_[instruction->out_] = 0; 
        }

        if (if_skip_this_sample) break;

        ++num_rank_op;
        ExecuteMyInstruction(*instruction, rand_gen_, &memory_, features);

        ++ins_count;            
      } else {
        ExecuteMyInstruction(*instruction, rand_gen_, &memory_, features);

        ++ins_count;
      }
    }

    // ProbabilityConverter<F>::Convert(&memory_);
    
    // Check whether we should stop early.
    const Scalar& label = train_it->GetLabel();
    if (std::abs(label) == 1234 || if_skip_this_sample) { /// first condition for missing data entry; second one see comments for if_skip_this_sample
      train_it->Next();
      if (train_it->Done()) {
        break;  // Reached the end of the dataset.
      }
      continue;
    }
    const double abs_error = ErrorComputer<F>::Compute(memory_, label);
    if (isnan(abs_error)) { // || abs_error > max_abs_error_
      return false;
    }
    if (errors != nullptr) {
      errors->push_back(abs_error);
    }

    // Run learn component function for this example.
    memory_.matrix_[kFeaturesMatrixAddress] = features;
    LabelAssigner<F>::Assign(label, &memory_);

    IntegerT ins_count_learn = 0;
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm_.learn_) {
      if ((*useful_list)[ins_count_learn] < 1) {
        ++ins_count_learn;
        continue;
      }
      ExecuteMyInstruction(*instruction, rand_gen_, &memory_, features);
      ++ins_count_learn;
    }
    // Check whether we are done.
    train_it->Next();
    if (train_it->Done()) {
      break;  // Reached the end of the dataset.
    }
  }
  return true;
}

template <FeatureIndexT F>
template <size_t max_component_function_size>
bool Executor<F>::TrainOptImpl(const IntegerT max_steps,
                               std::vector<double>* errors,
                               TaskIterator<F>* train_it, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_round, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list) {
  if (errors != nullptr) {
    errors->reserve(max_steps);
  }
  vector<double> relatiion_start_list = {0,36,69,74,79,235,237,369,386,387,406,411,452,464,508,524,530,539,543,551,553,570,599,605,616,622,635,643,661,663,670,671,691,700,701,707,718,721,724,726,730,732,734,744,749,757,764,767,775,782,802,805,809,813,816,822,825,830,833,837,843,850,852,860,862,866,869,875,880,882,890,891,896,898,903,904,908,910,915,921,928,940,946,952,954,957,964,965,966,971,975,979,984,992,994,995,997,1000,1003,1004,1005,1007,1008,1009,1010,1012,1014,1018,1020,1021,1022,1024,1025};

  if (dataset_.industry_relation_ == current_relation) {
    previous_relation = current_relation; // comparing the real current (dataset_.industry_relation_) with the actual previous (current_relation)
  } else current_relation_count = 0;
  current_relation = dataset_.industry_relation_;  
  ++current_relation_count;
  

  std::array<Instruction, max_component_function_size>
      optimized_predict_component_function;
  typename std::array<Instruction, max_component_function_size>::iterator
      optimized_predict_instr_it = optimized_predict_component_function.begin();
  for (const std::shared_ptr<const Instruction>& predict_instr :
       algorithm_.predict_) {
    *optimized_predict_instr_it = *predict_instr;
    ++optimized_predict_instr_it;
  }

  std::array<Instruction, max_component_function_size>
      optimized_learn_component_function;
  typename std::array<Instruction, max_component_function_size>::iterator
      optimized_learn_instr_it = optimized_learn_component_function.begin();
  for (const std::shared_ptr<const Instruction>& learn_instr :
       algorithm_.learn_) {
    *optimized_learn_instr_it = *learn_instr;
    ++optimized_learn_instr_it;
  }
  const IntegerT num_learn_instr = algorithm_.learn_.size();

  for (IntegerT step = 0; step < max_steps; ++step) {
    num_train_steps_completed_++;
    // Run predict component function for this example.
    const Matrix<F>& features = train_it->GetFeatures();

    if (CheckFeature(features)) {
      train_it->Next();
      if (train_it->Done()) {
        break;  // Reached the end of the dataset.
      }
      continue;
    }
    memory_.matrix_[kFeaturesMatrixAddress] = features;
    ZeroLabelAssigner<F>::Assign(&memory_);

    IntegerT ins_count = 0;

    IntegerT num_rank_op = 0;

    *num_stock_rank = 0;
    *num_TS_rank = 0;

    IntegerT num_stock_rank_count = 0;
    IntegerT num_TS_rank_count = 0;

    bool if_skip_this_sample = false; /// don't learn this sample because of rank op not enough data to compare yet

    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm_.predict_) {

      double out = 0;
      
      if ((*useful_list)[algorithm_.learn_.size() + ins_count] < 1) {
        ++ins_count;
        continue;
      }
      if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 73 || instruction->op_ == 75) {

        if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 75) {
          if(isnan(memory_.scalar_[instruction->in1_])) {
            ++ins_count;
            continue;
          }
          if (instruction->op_ == 66 || instruction->op_ == 75 || instruction->op_ == 65) ++num_stock_rank_count;
          if (instruction->op_ == 72) ++num_TS_rank_count;
        } else if (instruction->op_ == 73) {
          if(isnan(memory_.scalar_[instruction->in1_])) {
              ++ins_count;
              continue;
            }
          ++num_stock_rank_count;
          ++num_TS_rank_count;         
        }

      switch (instruction->op_) {
        case 75: {
          if (((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0) {    
            IntegerT relation_len;
            if (dataset_.industry_relation_ != 112) relation_len = relatiion_start_list[dataset_.industry_relation_ + 1] - relatiion_start_list[dataset_.industry_relation_];
            else relation_len = 1;
            if (relation_len > 5) {
              for (IntegerT for_compare = relatiion_start_list[dataset_.industry_relation_]; for_compare < relatiion_start_list[dataset_.industry_relation_ + 1]; ++for_compare) {
                out+=(*tasks_rank)[num_rank_op][for_compare][step];
              }                
            } else memory_.scalar_[instruction->out_] = 0;

            memory_.scalar_[instruction->out_] = memory_.scalar_[instruction->in1_] - out/relation_len;
          }

          if ((this_round + 1) == num_stock_rank_count) {
            (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
          }
          if(!(((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0)) if_skip_this_sample = true;

          break;  
        }
        case 65: {
          if (((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0) {  
            IntegerT relation_len;
            if (dataset_.industry_relation_ != 112) relation_len = relatiion_start_list[dataset_.industry_relation_ + 1] - relatiion_start_list[dataset_.industry_relation_];
            else relation_len = 1;
            if (relation_len > 5) {
              for (IntegerT for_compare = relatiion_start_list[dataset_.industry_relation_]; for_compare < relatiion_start_list[dataset_.industry_relation_ + 1]; ++for_compare) {
                if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][for_compare][step]) {
                  ++out;
                }                
              }

              memory_.scalar_[instruction->out_] = out/relation_len;
            } else memory_.scalar_[instruction->out_] = 0;
          }
          // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock
          if ((this_round + 1) == num_stock_rank_count) {
            (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
          }
          if(!((this_round + 1) > num_stock_rank_count && num_stock_rank_count > 0)) if_skip_this_sample = true;

          break;  
        }
        case 66: {
          ++(*num_stock_rank);
          
          if (this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) {
     
            for (IntegerT for_compare = 0; for_compare < num_of_stocks_to_approximate_rank; ++for_compare) {
              if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][for_compare][step]) {
                ++out;
              }

            }
            memory_.scalar_[instruction->out_] = out/(num_of_stocks_to_approximate_rank);
          }
          // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock
          if (this_round + 1 == num_stock_rank_count) (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
          if(!(this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0)) if_skip_this_sample = true;

          break;  
        }
        case 73: { 
          if ((this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) && step > (13 * (*num_TS_rank) + 12)) { /// if step is larger then 12, means there are 13 steps before you can compare    
            IntegerT count = 0;
            for (IntegerT for_compare_step = step - 1 ; for_compare_step > (step - 2 - 12); --for_compare_step) { /// limit case is 0 to 12
              double count_out = 0.0;
              for (IntegerT for_compare = 0; for_compare < num_of_stocks_to_approximate_rank; ++for_compare) {
                if ((*tasks_rank)[num_rank_op][task_index][for_compare_step] > (*tasks_rank)[num_rank_op][for_compare][for_compare_step]) ++count_out;
              }
              memory_.vector_[instruction->out_](count) = count_out/num_of_stocks_to_approximate_rank;
              ++count;
            }
          }
          // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock e.g. -1 < task_index < 10 or 9 < task_index < 20 or  19 < task_index < 30 ... 
          if ((this_round + 1 == num_stock_rank_count) && (13 * ((*num_TS_rank) - 1) + 12) < step) {
            (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
          }
          if (!((this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) && step > (13 * (*num_TS_rank) + 12))) if_skip_this_sample = true;

          ++(*num_stock_rank);
          ++(*num_TS_rank);        

          break;
        }
        case 72: {

          if (step > (13 * (*num_TS_rank) + 12)) {
            for (IntegerT for_compare = step - 1; for_compare > (step - 2 - 12); --for_compare) {
              if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][task_index][for_compare]) ++out;
              }
              memory_.scalar_[instruction->out_] = out/13;    
            }
          if ((13 * ((*num_TS_rank) - 1) + 12) < step) {
            (*tasks_rank)[num_rank_op][task_index][step] = memory_.scalar_[instruction->in1_];
          }
          if (!(step > (13 * (*num_TS_rank) + 12))) if_skip_this_sample = true;

          ++(*num_TS_rank);        

          break;
        }  
        default:
          memory_.scalar_[instruction->out_] = 0; 
      }

      if (if_skip_this_sample) break;

      ++num_rank_op;
      ExecuteMyInstruction(*instruction, rand_gen_, &memory_, features);
      ++ins_count;            
     } else {
      ExecuteMyInstruction(*instruction, rand_gen_, &memory_, features);
      ++ins_count;
     }
    }

    // allow all to sigmod because predicting return
    // ProbabilityConverter<F>::Convert(&memory_);
    // Check whether we should stop early.
    const Scalar& label = train_it->GetLabel();

    if (std::abs(label) == 1234 || if_skip_this_sample) {
      train_it->Next();
      if (train_it->Done()) {
        break;  // Reached the end of the dataset.
      }
      continue;
    }
    const double abs_error = ErrorComputer<F>::Compute(memory_, label);
    if (isnan(abs_error)) { //  || abs_error > max_abs_error_
      return false;
    }
    if (errors != nullptr) {
      errors->push_back(abs_error);
    }
    // Run learn component function for this example.
    memory_.matrix_[kFeaturesMatrixAddress] = features;
    if (CheckFeature(features)) {
      train_it->Next();
      if (train_it->Done()) {
        break;  // Reached the end of the dataset.
      }
      continue;
    }
    LabelAssigner<F>::Assign(label, &memory_);
    IntegerT learn_instr_num = 0;
    for (const Instruction& instruction : optimized_learn_component_function) {
      if (learn_instr_num == num_learn_instr) {
        break;
      }
      if ((*useful_list)[learn_instr_num] < 1) {
        ++learn_instr_num;
        continue;
      }      
      ExecuteMyInstruction(instruction, rand_gen_, &memory_, features);
      ++learn_instr_num;
    }

    // Check whether we are done.
    train_it->Next();
    if (train_it->Done()) {
      break;  // Reached the end of the dataset.
    }
  }
  return true;
}

// Minimum negative error tolerated to account for numerical issue around zero.
constexpr double kNegativeErrorTolerance = -1e-6;

template <FeatureIndexT F>
struct SquashedRmseLossAccumulator {
  inline static void Accumulate(
      const Memory<F>& memory, const Scalar& label,
      double* error, double* loss, double* pred) {
    *pred = 2 * Sigmoid(memory.scalar_[kPredictionsScalarAddress]) - 1;
    *error = label - *pred;
    if (std::abs(*error) > 4) {
      "here here label is wrong!!!";
      cout << "label" << label << endl;
      cout << "*pred" << *pred << endl;
    }
    *loss += *error * *error;
  }
};

template <FeatureIndexT F>
struct ProbAccuracyLossAccumulator {
  inline static void Accumulate(
      const Memory<F>& memory, const Scalar& label,
      double* error, double* loss, double* pred) {
    double logit = memory.scalar_[kPredictionsScalarAddress];
    double pred_prob = Sigmoid(logit);
    *pred = pred_prob; 
    // added isnan(pred_prob) to below because nan compare with 0.5 would return false than is_correct is true. This would assign no loss.
    if ((pred_prob > 1.0) || (pred_prob < 0.0) || isnan(pred_prob)) {
      *error = std::numeric_limits<double>::infinity();
    } else {
      bool is_correct = ((label > 0.5) == (pred_prob > 0.5));
      *error = is_correct ? 0.0 : 1.0;
    }
    *loss += *error;
  }
};

template <FeatureIndexT F>
double Executor<F>::Validate(const IntegerT max_steps, std::vector<double>* errors, std::vector<double>* preds, std::vector<double>* price_diff, std::vector<std::vector<std::vector<double>>>* tasks_rank, IntegerT this_round, IntegerT task_index, IntegerT* num_stock_rank, IntegerT* num_TS_rank, const IntegerT num_of_stocks_to_approximate_rank, std::vector<IntegerT> *useful_list) {
  double loss = 0.0;
  double skip_sample = 0;
  if (errors != nullptr) {
    errors->reserve(dataset_.ValidSteps());
  }
  if (preds != nullptr) {
    preds->reserve(dataset_.ValidSteps());
  }
  if (price_diff != nullptr) {
    price_diff->reserve(dataset_.ValidSteps());
  }

  vector<double> relatiion_start_list = {0,36,69,74,79,235,237,369,386,387,406,411,452,464,508,524,530,539,543,551,553,570,599,605,616,622,635,643,661,663,670,671,691,700,701,707,718,721,724,726,730,732,734,744,749,757,764,767,775,782,802,805,809,813,816,822,825,830,833,837,843,850,852,860,862,866,869,875,880,882,890,891,896,898,903,904,908,910,915,921,928,940,946,952,954,957,964,965,966,971,975,979,984,992,994,995,997,1000,1003,1004,1005,1007,1008,1009,1010,1012,1014,1018,1020,1021,1022,1024,1025};

  const IntegerT num_steps =
      std::min(num_valid_examples_,
               static_cast<IntegerT>(dataset_.ValidSteps()));
  CHECK(errors == nullptr || num_steps <= 100) <<
      "You should only record the validation errors for few validation steps."
      << std::endl;

  TaskIterator<F> valid_it = dataset_.ValidIterator();

  // std::string filename_2 = "/hdd8/james/stock_prediction_system/AlphaEvolve_HF/debug.txt";
  // std::ofstream outFile(filename_2, std::ios::app);

  // outFile << "task_index: " << task_index << "\n";

  for (IntegerT step = 0; step < num_steps; ++step) {
    // if (step > 440) outFile << "num_steps" << step << "\n";
    // Run predict component function for this example.
    const Matrix<F>& features = valid_it.GetFeatures();
    if (CheckFeature(features)) {
      if (preds != nullptr) {
        preds->push_back(-1234);
      }
      if (price_diff != nullptr) {
        price_diff->push_back(-1234);
      }

      ++skip_sample;
      valid_it.Next();

      continue;
    }
    memory_.matrix_[kFeaturesMatrixAddress] = features;
    ZeroLabelAssigner<F>::Assign(&memory_);

    IntegerT ins_count = 0;

    IntegerT num_rank_op = 0;

    *num_stock_rank = 0;
    *num_TS_rank = 0;

    IntegerT num_stock_rank_count = 0;
    IntegerT num_TS_rank_count = 0;

    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm_.predict_) {

      double out = 0;
      
      if ((*useful_list)[algorithm_.learn_.size() + ins_count] < 1) {
        ++ins_count;
        continue;
      }
      if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 73 || instruction->op_ == 75) {

        if (instruction->op_ == 65 || instruction->op_ == 66 || instruction->op_ == 72 || instruction->op_ == 75) {
          if(isnan(memory_.scalar_[instruction->in1_])) {
            ++ins_count;
            continue;
          }
          if (instruction->op_ == 66 || instruction->op_ == 75 || instruction->op_ == 65) ++num_stock_rank_count;
          if (instruction->op_ == 72) ++num_TS_rank_count;
        } else if (instruction->op_ == 73) {
          if(isnan(memory_.scalar_[instruction->in1_])) {
              ++ins_count;
              continue;
            }
          ++num_stock_rank_count;
          ++num_TS_rank_count;         
        }
      // std::cout << "instruction->op_" << instruction->op_ << std::endl;
      switch (instruction->op_) {
        case 75: {
          if (((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0) {     
            IntegerT relation_len;
            if (dataset_.industry_relation_ != 112) relation_len = relatiion_start_list[dataset_.industry_relation_ + 1] - relatiion_start_list[dataset_.industry_relation_];
            else relation_len = 1;
            if (relation_len > 5) {
              for (IntegerT for_compare = relatiion_start_list[dataset_.industry_relation_]; for_compare < relatiion_start_list[dataset_.industry_relation_ + 1]; ++for_compare) {
                out+=(*tasks_rank)[num_rank_op][for_compare][step + max_steps];
              }                
            } else memory_.scalar_[instruction->out_] = 0;

            memory_.scalar_[instruction->out_] = memory_.scalar_[instruction->in1_] - out/relation_len;
          }

          if ((this_round + 1) == num_stock_rank_count) {
            (*tasks_rank)[num_rank_op][task_index][step + max_steps] = memory_.scalar_[instruction->in1_];
          }

          break;  
        }
        case 65: {
          if (((this_round + 1) > num_stock_rank_count) && num_stock_rank_count > 0) {  
            IntegerT relation_len;
            if (dataset_.industry_relation_ != 112) relation_len = relatiion_start_list[dataset_.industry_relation_ + 1] - relatiion_start_list[dataset_.industry_relation_];
            else relation_len = 1;
            if (relation_len > 5) {
              for (IntegerT for_compare = relatiion_start_list[dataset_.industry_relation_]; for_compare < relatiion_start_list[dataset_.industry_relation_ + 1]; ++for_compare) {
                if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][for_compare][step + max_steps]) {
                  ++out;
                }
              }

              memory_.scalar_[instruction->out_] = out/relation_len;
            } else memory_.scalar_[instruction->out_] = 0;
          }
          // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock
          if ((this_round + 1) == num_stock_rank_count) {
            (*tasks_rank)[num_rank_op][task_index][step + max_steps] = memory_.scalar_[instruction->in1_];
          }
          break;  
        }
        case 66: {
          ++(*num_stock_rank);
          
          if (this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) { 
            for (IntegerT for_compare = 0; for_compare < num_of_stocks_to_approximate_rank; ++for_compare) {
              if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][for_compare][step + max_steps]) {
                ++out;
              }
            }
            memory_.scalar_[instruction->out_] = out/(num_of_stocks_to_approximate_rank);
          }
          // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock
          if (this_round + 1 == num_stock_rank_count) (*tasks_rank)[num_rank_op][task_index][step + max_steps] = memory_.scalar_[instruction->in1_];
          break;  
        }
        case 73: { 
          if ((this_round + 1 > num_stock_rank_count && num_stock_rank_count > 0) && step > (13 * (*num_TS_rank) + 12)) { /// if step is larger then 12, means there are 13 steps before you can compare     
            IntegerT count = 0;
            for (IntegerT for_compare_step = step - 1 ; for_compare_step > (step - 2 - 12); --for_compare_step) { /// limit case is 0 to 12
              double count_out = 0.0;
              for (IntegerT for_compare = 0; for_compare < num_of_stocks_to_approximate_rank; ++for_compare) {
                if ((*tasks_rank)[num_rank_op][task_index][for_compare_step + max_steps] > (*tasks_rank)[num_rank_op][for_compare][for_compare_step + max_steps]) ++count_out;
              }
              memory_.vector_[instruction->out_](count) = count_out/num_of_stocks_to_approximate_rank;
              ++count;
            }
          }

          // save in1 of current num_rank_op when it can't be read; put behind the previous if because we cannot change what to compare before we compare it after task index of 9. to rule to random substitute a stock e.g. -1 < task_index < 10 or 9 < task_index < 20 or  19 < task_index < 30 ... 
          if ((this_round + 1 == num_stock_rank_count) && (13 * ((*num_TS_rank) - 1) + 12) < step) {
            (*tasks_rank)[num_rank_op][task_index][step + max_steps] = memory_.scalar_[instruction->in1_];
          }
          ++(*num_stock_rank);
          ++(*num_TS_rank);        

          break;
        }
        case 72: {

          if (step > (13 * (*num_TS_rank) + 12)) {
            for (IntegerT for_compare = step - 1; for_compare > (step - 2 - 12); --for_compare) {
              if (memory_.scalar_[instruction->in1_] > (*tasks_rank)[num_rank_op][task_index][for_compare]) ++out;
              }
              memory_.scalar_[instruction->out_] = out/13;    
            }
          if ((13 * ((*num_TS_rank) - 1) + 12) < step) {
            (*tasks_rank)[num_rank_op][task_index][step + max_steps] = memory_.scalar_[instruction->in1_];
          }

          ++(*num_TS_rank);        

          break;
        }  
        default:
          memory_.scalar_[instruction->out_] = 0; 
      }
      ++num_rank_op;
      ExecuteMyInstruction(*instruction, rand_gen_, &memory_, features);
      ++ins_count;            
     } else {
     
      // if (step > 440)outFile <<  "Memory before ExecuteMyInstruction:\n";
      // for (const auto& val : memory_.matrix_) {
      //     if (step > 440) outFile <<  val << " "<< "\n";
      //     break;
      // }
      // if (step > 440) outFile <<  "scalars: ";
      // for (const auto& val : memory_.scalar_) {
      //     if (step > 440) outFile <<  val << " ";
      // }      
      // if (step > 440) outFile <<  "\n";
      // if (step > 440)outFile <<  "instruction" << instruction<< "\n"; 
      // if (step > 440)outFile <<  "instruction->op_ " << instruction->op_<< "\n";
      // if (step > 440)outFile <<  "instruction->in1_ " << instruction->in1_<< " " << memory_.scalar_[instruction->in1_]<< "\n";
      // if (step > 440)outFile <<  "instruction->in2_ " << instruction->in2_<< " " << memory_.scalar_[instruction->in2_]<< "\n";
      // if (step > 440)outFile <<  "instruction->out_ " << instruction->out_<< " " << memory_.scalar_[instruction->out_]<< "\n";
      ExecuteMyInstruction(*instruction, rand_gen_, &memory_, features);
      // if (step > 440)outFile <<  "instruction->op_ " << instruction->op_<< "\n";
      // if (step > 440)outFile <<  "instruction->in1_ " << instruction->in1_<< " " << memory_.scalar_[instruction->in1_]<< "\n";
      // if (step > 440)outFile <<  "instruction->in2_ " << instruction->in2_<< " " << memory_.scalar_[instruction->in2_]<< "\n";
      // if (step > 440)outFile <<  "instruction->out_ " << instruction->out_<< " " << memory_.scalar_[instruction->out_]<< "\n";      
      // if (step > 440)outFile <<  "Memory after ExecuteMyInstruction:\n";
      // for (const auto& val : memory_.matrix_) {
      //     std::cout << val << " ";
      //     break;
      // }         

      ++ins_count;
     }
    }
    // ProbabilityConverter<F>::Convert(&memory_);
    // if (step > 440)outFile << "after scalars: "<< "\n";
    // for (const auto& val : memory_.scalar_) {
    //     if (step > 440)outFile << val << " ";
    // }    
    // Accumulate the loss.
    double error = 0.0;
    double pred = 0.0;
    const Scalar& label = valid_it.GetLabel();
    // if (step > 440)outFile <<  "label" << label<< "\n"; 
    if (std::abs(label) == 1234 || std::abs(memory_.scalar_[kPredictionsScalarAddress]) < 1e-01) {

      if (preds != nullptr) {
        preds->push_back(-1234);
      }
      if (price_diff != nullptr) {
        price_diff->push_back(-1234);
      }

      ++skip_sample;
      valid_it.Next();
      // if (step > 440)outFile <<  "invalid memory_.scalar_[kPredictionsScalarAddress]" << memory_.scalar_[kPredictionsScalarAddress] << "\n"; 
      continue;
    }
    switch (dataset_.eval_type_) {
      case RMS_ERROR: {
        SquashedRmseLossAccumulator<F>::Accumulate(memory_, label, &error,
                                                   &loss, &pred);
        break;
      }
      case ACCURACY: {
        ProbAccuracyLossAccumulator<F>::Accumulate(memory_, label, &error,
                                                   &loss, &pred);
        break;
      }
      case NO_CHANGE: {
        pred = memory_.scalar_[kPredictionsScalarAddress];
        break;
      }
      case INVALID_EVAL_TYPE:
        LOG(FATAL) << "Invalid eval type." << std::endl;
      // Do not add default case here. All enum values should be supported.
    }
    const double abs_error = std::abs(error);
     
    if (isnan(abs_error) || isnan(pred)) { // || abs_error > max_abs_error_ 

      if (preds != nullptr && price_diff != nullptr) {
        preds->clear();
        price_diff->clear();
      }

      // Stop early. Return infinite loss.
      return kMinFitness;
    }

    if (errors != nullptr) {
      errors->push_back(std::abs(error));
    }
    if (preds != nullptr) {
      preds->push_back(pred);
    } 
    if (price_diff != nullptr) {
      price_diff->push_back(valid_it.GetLabel());
    }
    // if (step > 440)outFile <<  "max_abs_error_: " << max_abs_error_ << " " << "\n";
    // if (step > 440)outFile <<  "pred into preds: " << pred << " " << "\n";
    // if (step > 440)outFile <<  "into price_diff vec valid_it.GetLabel() " << valid_it.GetLabel() << " " << "\n";
    valid_it.Next();
  }
    
  // outFile.close(); 
  // Convert to fitness.
  double fitness;
  switch (dataset_.eval_type_) {
    case INVALID_EVAL_TYPE:
      LOG(FATAL) << "Invalid eval type." << std::endl;
    case RMS_ERROR:
      loss /= (static_cast<double>(dataset_.ValidSteps()) - static_cast<double>(skip_sample));
      fitness = FlipAndSquash(sqrt(loss));
      break;
    case ACCURACY:
      loss /= (static_cast<double>(dataset_.ValidSteps()) - static_cast<double>(skip_sample));
      fitness = 1.0 - loss;
      break;
  }

  return fitness;
}

template <FeatureIndexT F>
void Executor<F>::GetMemory(Memory<F>* memory) {
  memory->scalar_ = memory_.scalar_;
  memory->vector_ = memory_.vector_;
  memory->matrix_ = memory_.matrix_;
}

template <FeatureIndexT F>
void ExecuteAndFillLabels(const Algorithm& algorithm, Memory<F>* memory,
                          TaskBuffer<F>* buffer,
                          RandomGenerator* rand_gen) {
  // Fill training labels.
  typename std::vector<Scalar>::iterator train_label_it =
      buffer->train_labels_.begin();
  for (const Matrix<F>& train_features : buffer->train_features_) {
    // Run predict component function for this example.
    memory->matrix_[kFeaturesVectorAddress] = train_features;
    ZeroLabelAssigner<F>::Assign(memory);
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm.predict_) {
      ExecuteInstruction(*instruction, rand_gen, memory);
    }
    *train_label_it = PredictionGetter<F>::Get(memory);
    ++train_label_it;
  }

  // Fill validation labels.
  std::vector<Scalar>::iterator valid_label_it =
      buffer->valid_labels_.begin();
  for (const Matrix<F>& valid_features : buffer->valid_features_) {
    // Run predict component function for this example.
    memory->matrix_[kFeaturesVectorAddress] = valid_features;
    ZeroLabelAssigner<F>::Assign(memory);
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm.predict_) {
      ExecuteInstruction(*instruction, rand_gen, memory);
    }
    *valid_label_it = PredictionGetter<F>::Get(memory);
    ++valid_label_it;
  }
}

constexpr double kMinusTwoOverPi = -0.63661977236758138243;

inline double FlipAndSquash(const double value) {
  if (isnan(value) || isinf(value)) {
    return 0.0;
  }
  CHECK_GE(value, 0.0);
  return static_cast<double>(1.0) + kMinusTwoOverPi * atan(value);
}

namespace internal {

template<FeatureIndexT F>
inline Vector<F> TruncatingSoftmax(const Vector<F>& input) {
  // TODO(ereal): rewrite using Eigen's block<>() method.
  // TODO(ereal): consider reusing vectors.
  Vector<kNumClasses> truncated;
  truncated.resize(kNumClasses, 1);
  for (FeatureIndexT i = 0; i < kNumClasses; ++i) {
    truncated(i) = input(i);
  }
  const Vector<kNumClasses> shifted =
      truncated - Vector<kNumClasses>::Ones(kNumClasses) * truncated.maxCoeff();
  const Vector<kNumClasses> exponentiated = shifted.array().exp().matrix();
  const double total = exponentiated.sum();
  const Vector<kNumClasses> normalized = exponentiated / total;
  Vector<F> padded;
  padded.resize(F, 1);
  for (FeatureIndexT i = 0; i < kNumClasses; ++i) {
    padded(i) = normalized(i);
  }
  for (FeatureIndexT i = kNumClasses; i < F; ++i) {
    padded(i) = kPadLabel;
  }
  return padded;
}

template<FeatureIndexT F>
inline FeatureIndexT Argmax(const Vector<F>& input) {
  FeatureIndexT max_index = 0;
  double max_element = std::numeric_limits<double>::lowest();
  for (FeatureIndexT index = 0; index < F; ++index) {
    if (input(index) >= max_element) {
      max_index = index;
      max_element = input(index);
    }
  }
  return max_index;
}

// Computes -x * log(y). Assumes 0.0 <= x,y <= 1.0.
inline double MinusXLogY(const double x, const double y) {
  constexpr double zero = 0.0;
  if (y > zero) {
    return - x * log(y);
  } else {
    if (x > zero) {
      return std::numeric_limits<double>::infinity();
    } else {
      return zero;
    }
  }
}

}  // namespace internal

}  // namespace alphaevolve

#endif  // EXECUTOR_H_

